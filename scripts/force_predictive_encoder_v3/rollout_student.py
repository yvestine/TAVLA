"""Continue the frozen Real teacher with native-clock, source-mapped Sim rollouts.

Real labels use original reference frame offsets +1..+5. Explicit recorded
source indices supply correspondence; force curves are never used to align data.
The legacy peg export's documented sign flip is undone to use incoming-link7
force in the Sim branch, as in the three new rollout tasks. No Real weights or
target statistics are changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import segment_limits
from . import BLOCK, FORMAT, FUTURE, HISTORY, LAYOUT, PAST
from .losses import objective
from .metrics import aggregate, reconstruction_metrics, rmse
from .multitask import atomic_torch_save, emit, source_identity
from .multitask_data import ForceEpisode, data_identity, fit_domain_stats, load_real, prepare_arrays, task_config
from .train import cpu_state, digest, load_model, save_model, seed_all

TASK_NAMES = {'dual_peg_insert': 'dual_peg_insert_right', 'plate_slide': 'plate_sliding_right',
              'usb_insert': 'usb-insertio_right'}


def interpolate_reference(values, index, starts):
    """Interpolate supervision only inside a continuous original recording segment."""
    index = np.asarray(index, np.float64)
    finite = np.isfinite(index)
    clipped = np.clip(np.where(finite, index, 0), 0, len(values)-1)
    lo, hi = np.floor(clipped).astype(int), np.ceil(clipped).astype(int)
    valid = finite & (index >= 0) & (index <= len(values)-1) & (starts[lo] == starts[hi])
    fraction = clipped-lo
    while fraction.ndim < values[lo].ndim: fraction = fraction[..., None]
    return (values[lo]*(1-fraction) + values[hi]*fraction).astype(np.float32), valid


def supervision(real, source_index):
    positions = source_index[:, None] + np.arange(FUTURE+1)[None]
    force, valid = interpolate_reference(real.force, positions, real.starts)
    base = np.floor(np.clip(source_index, 0, len(real.force)-1)).astype(int)
    target = np.floor(np.clip(positions, 0, len(real.force)-1)).astype(int)
    valid &= real.starts[target] == real.starts[base, None]
    valid &= valid[:, :1]
    return dict(current=force[:, 0], future=force[:, 1:], valid=valid[:, 0], future_mask=valid[:, 1:])


def load_pairs(project, teacher_run, bundle):
    recipe = json.loads((teacher_run/'recipe.json').read_text())
    manifest = json.loads((teacher_run/'manifest.json').read_text())
    if manifest.get('ready_domains') != ['real'] or recipe.get('stage') != 'teacher':
        raise ValueError('A completed Real teacher is required')
    if recipe['source_sha256'] != source_identity(): raise ValueError('Original teacher source changed')
    tasks = task_config(project/'configs/force_predictive_v3_host9_multitask_workspace.json', project)
    real = load_real(tasks)
    if data_identity(real) != recipe['real_data']: raise ValueError('Real teacher recordings changed')
    checkpoint = teacher_run/manifest['checkpoint']
    if sha256(checkpoint) != manifest['encoder_sha256']: raise ValueError('Teacher checkpoint hash mismatch')
    pairs, audit = {}, {}
    def add(key, path, force, clock, index, contract):
        if key not in real or key in pairs: raise ValueError(f'Unknown/duplicate pair: {key}')
        if force.shape != (len(clock),3) or index.shape != clock.shape or len(clock)<PAST:
            raise ValueError(f'{path}: invalid force/time/source-index shape')
        if not all(np.isfinite(x).all() for x in (force,clock,index)) or np.any(np.diff(clock)<=0) or np.any(np.diff(index)<-1e-7):
            raise ValueError(f'{path}: invalid time or reference mapping')
        if index.min() < -1e-7 or index.max() > len(real[key].force)-1+1e-7:
            raise ValueError(f'{path}: source mapping outside Real recording')
        index = np.clip(index,0,len(real[key].force)-1)
        starts,ends = segment_limits(clock)
        task,tid=key.rsplit('/traj_',1)
        native=ForceEpisode(task,int(tid),force.astype(np.float32),clock,starts,ends,path,sha256(path))
        target=supervision(real[key],index)
        pairs[key]=dict(ep=native,index=index,**target)
        audit[key]=dict(path=str(path),sha256=native.sha256,real_sha256=real[key].sha256,
                        frames=len(clock),valid_reference_frames=int(target['valid'].sum()),
                        valid_future_labels=int(target['future_mask'].sum()),contract=contract)
    for path in sorted((project/'sim-data-aligned').glob('traj_*/data.h5')):
        key=f'peg-in-hole/{path.parent.name}'
        with h5py.File(path,'r') as h:
            f=h['obs/state/ee_force_base'][:]; t=h['timestamps'][:]
            if 'sign=-1,scale=1.000' not in str(h.attrs.get('force_alignment','')):
                raise ValueError(f'{path}: unexpected legacy force definition')
        if len(t)!=len(real[key].time) or not np.allclose(t,real[key].time,atol=1e-6,rtol=0):
            raise ValueError(f'{path}: legacy reference times differ')
        add(key,path,-f,t,np.arange(len(t),dtype=float),'legacy calibrated incoming link7 = -aligned XYZ')
    package=json.loads((bundle/'training_manifest.json').read_text())
    for entry in package['episodes']:
        if entry['state']!='eligible': continue
        key=f"{TASK_NAMES[entry['task']]}/{entry['trajectory']}"
        path=bundle/entry['data_h5']
        if sha256(path)!=entry['data_h5_sha256'] or sha256(bundle/entry['source'])!=real[key].sha256:
            raise ValueError(f'{path}: rollout or source hash mismatch')
        with h5py.File(path,'r') as h:
            f=h['observations/effort'][:,:3]; t=h['timestamp'][:]
            rt=h['recorded_reference/times'][:]; index=h['recorded_reference/source_frame_index'][:]
            original=h['recorded_reference/original_times'][:]
            wrench=h['recorded_reference/original_wrench'][:,:3]
            if len(rt)!=len(t)+1 or not np.allclose(rt[:-1],t,atol=1e-6,rtol=0):
                raise ValueError(f'{path}: native observation/reference clocks differ')
            if original.shape!=real[key].time.shape or not np.allclose(original,real[key].time-real[key].time[0],atol=1e-6,rtol=0):
                raise ValueError(f'{path}: original reference times differ')
            if wrench.shape!=real[key].force.shape or not np.allclose(wrench,real[key].force,atol=1e-6):
                raise ValueError(f'{path}: original reference forces differ')
        add(key,path,f,t,index[:-1],'exported startup-tared incoming link7 XYZ, raw sign retained')
    split=json.loads((teacher_run/'split.json').read_text())
    split={**split, **{s:[k for k in split[s] if k in pairs] for s in ('train','validation')}}
    if not split['train'] or not split['validation']: raise ValueError('Empty inherited paired split')
    return real,pairs,split,audit,manifest


class MappedBlocks(Dataset):
    def __init__(self,pairs,keys,arrays,teacher_z,threshold):
        self.pairs,self.arrays,self.teacher_z=pairs,arrays,teacher_z
        counts={p['ep'].task:0 for k,p in pairs.items() if k in keys}
        for key in keys: counts[pairs[key]['ep'].task]+=1
        self.keys=[]; weights=[]
        for key in keys:
            p=pairs[key]; valid=np.flatnonzero(p['valid']); delta=np.linalg.norm(np.diff(p['current'],axis=0,prepend=p['current'][:1]),axis=1)
            event=np.minimum(4,1+delta[valid]/threshold)
            w=(.5/len(valid)+.5*event/event.sum())/(len(counts)*counts[p['ep'].task])
            self.keys.extend((key,int(t)) for t in valid); weights.extend(w)
        self.weights=torch.as_tensor(weights,dtype=torch.double)
    def __len__(self): return len(self.keys)
    def __getitem__(self,item):
        key,anchor=self.keys[item]; p=self.pairs[key]; ep=p['ep']; start=ep.starts[anchor]
        original=np.arange(anchor-BLOCK,anchor+1); ii=np.maximum(original,start)
        valid=(original>=start)&p['valid'][ii]
        return dict(x=self.arrays[key]['x'][ii],components=self.arrays[key]['components'][ii][None],
                    teacher_z=self.teacher_z[key][ii],future=p['future'][ii],
                    valid=valid,past_mask=(ii[:,None]+np.arange(-PAST+1,1)>=start)&valid[:,None],
                    future_mask=p['future_mask'][ii]&valid[:,None])


@torch.inference_mode()
def predict(model,arrays,domain,device):
    rows={}
    for lo in range(0,len(arrays['x']),128):
        out=model(torch.from_numpy(arrays['x'][lo:lo+128]).to(device),domain)
        for name,value in out.items(): rows.setdefault(name,[]).append(value.cpu().numpy())
    return {name:np.concatenate(parts) for name,parts in rows.items()}


@torch.inference_mode()
def evaluate(model,pairs,keys,arrays,stats,device,save=None):
    model.eval(); tasks={}
    for key in keys:
        p=pairs[key]; ep=p['ep']; out=predict(model,arrays[key],1,device)
        history=out['reconstruction']*np.asarray(stats['sim']['scale'])+np.asarray(stats['sim']['mean'])
        row=reconstruction_metrics(history,ep,stats,'sim')
        for k in range(FUTURE):
            mask=p['future_mask'][:,k]; error=out['future'][mask,k]-p['future'][mask,k]
            row[f'future_{k+1}_rmse_n']=rmse(error)
        row['forecast_score_n']=float(np.dot([row[f'future_{k}_rmse_n'] for k in (1,3,5)],[.25,.35,.4]))
        cov=np.cov(out['z'],rowvar=False); ev=np.maximum(np.linalg.eigvalsh(cov),0); prob=ev/max(ev.sum(),1e-12)
        row['latent_effective_rank']=float(np.exp(-(prob*np.log(prob+1e-12)).sum()))
        tasks.setdefault(ep.task,{})[str(ep.trajectory)]=row
        if save is not None:
            path=save/ep.task/f'traj_{ep.trajectory}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**out,time=ep.time,source_frame_index=p['index'],target_future=p['future'],future_mask=p['future_mask'])
    by_task={task:aggregate(list(rows.values())) for task,rows in tasks.items()}
    return dict(summary=aggregate(list(by_task.values())),by_task=by_task,trajectories=tasks)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--teacher-run',type=Path,default=Path('eval_outputs/force_predictive_v3_multitask/host9_teacher'))
    p.add_argument('--bundle',type=Path,default=Path('host9_rollouts_20261026_105valid'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-epochs',type=int,default=200); p.add_argument('--patience',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--align-weight',type=float,default=.1); p.add_argument('--resume',action='store_true')
    p.add_argument('--audit-only',action='store_true'); a=p.parse_args(argv)
    for name in ('project','teacher_run','bundle','output_dir'): setattr(a,name,getattr(a,name).resolve())
    if min(a.threads,a.max_epochs,a.patience,a.batch_size)<1: p.error('positive integer arguments required')
    torch.set_num_threads(a.threads)
    real,pairs,split,audit,parent_manifest=load_pairs(a.project,a.teacher_run,a.bundle)
    model,_=load_model(a.teacher_run/parent_manifest['checkpoint'])
    stats=copy.deepcopy(model.stats)
    native={k:v['ep'] for k,v in pairs.items()}
    stats['sim'],sim_threshold=fit_domain_stats(native,split['train'])
    stats['event_threshold_n']['sim']=sim_threshold; model.stats=copy.deepcopy(stats)
    recipe=dict(format='force_predictive_v3_native_rollout_student_v1',parent_teacher_sha256=parent_manifest['encoder_sha256'],
                split=split,sim_data=audit,normalization=stats,source_sha256=source_identity(),
                adapter_sha256=sha256(Path(__file__)),hyperparameters={k:getattr(a,k) for k in ('max_epochs','patience','batch_size','lr','align_weight')},
                supervision='original_real_frame_offsets_1_to_5_at_recorded_source_index; native_sim_clock; no_force_curve_alignment')
    out=a.output_dir
    if (out/'recipe.json').exists():
        if not a.resume or json.loads((out/'recipe.json').read_text())!=recipe: raise ValueError('Use a new output directory or identical resume recipe')
    elif out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'recipe.json',recipe); write_json(out/'split.json',split); write_json(out/'normalization.json',stats)
    write_json(out/'data_audit.json',dict(episodes=audit,excluded_real_keys=sorted(set(real)-set(pairs)),
                                       paired_episodes=len(pairs),frames=sum(len(x.force) for x in native.values())))
    emit(out,f"Paired Sim: {len(pairs)} episodes; train={len(split['train'])} validation={len(split['validation'])}; native 10Hz input, source-mapped Real targets")
    if a.audit_only: return
    if a.resume and (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        if sha256(out/m['checkpoint'])!=m['encoder_sha256']: raise ValueError('Completed encoder hash mismatch')
        emit(out,'Reusing completed Sim encoder'); return
    seed_all(split['seed']+1000); model.initialize_student(); model.configure('student'); model.to(a.device)
    frozen=digest(model,frozen_only=True)
    arrays=prepare_arrays(native,stats,'sim'); teacher_z={}
    emit(out,'Preparing frozen Real teacher targets')
    real_arrays=prepare_arrays(real,stats,'real')
    for key,pair in pairs.items():
        z=predict(model,real_arrays[key],0,a.device)['z']
        teacher_z[key]=interpolate_reference(z,pair['index'],real[key].starts)[0]
    del real_arrays
    data=MappedBlocks(pairs,split['train'],arrays,teacher_z,stats['event_threshold_n']['real'])
    steps=max(1,math.ceil(len(data)/(a.batch_size*BLOCK)))
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=a.lr,weight_decay=1e-4)
    initial=evaluate(model,pairs,split['validation'],arrays,stats,a.device)
    score=lambda m: m['summary']['forecast_score_n']/float(np.mean(stats['target']['scale']))+.25*m['summary']['reconstruction_nrmse']
    initial_score=score(initial); write_json(out/'initial_metrics.json',initial)
    records=[]; best_score=math.inf; best_state=None; best_epoch=0; stale=0; first=1; elapsed_before=0
    if a.resume and (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        records=saved['records']; best_score=saved['best_score']; best_state=saved['best_state']; best_epoch=saved['best_epoch']; stale=saved['stale']; first=saved['epoch']+1; elapsed_before=saved['elapsed_s']
        torch.set_rng_state(saved['rng_state'])
        if a.device.startswith('cuda'): torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        if digest(model,frozen_only=True)!=frozen: raise ValueError('Frozen teacher changed in checkpoint')
    started=time.monotonic(); emit(out,f'Initial validation score={initial_score:.6f}; steps_per_epoch={steps}; starting epoch {first}')
    for epoch in range(first,a.max_epochs+1):
        if stale>=a.patience: break
        model.train(); losses=[]
        sampler=WeightedRandomSampler(data.weights,steps*a.batch_size,True,generator=torch.Generator().manual_seed(split['seed']+epoch*7919))
        for raw in DataLoader(data,batch_size=a.batch_size,sampler=sampler,num_workers=0):
            batch={k:v.to(a.device) for k,v in raw.items()}; optimizer.zero_grad(set_to_none=True)
            x=batch['x']; b,k=x.shape[:2]
            output={name:v.reshape(b,k,*v.shape[1:]) for name,v in model(x.reshape(b*k,*x.shape[2:]),1).items()}
            # The only reconstruction-target slot is Sim; encoder branch 1 was selected above.
            loss,detail=objective(model,output,batch,0,a.align_weight,{'z':batch['teacher_z']})
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite student loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3,error_if_nonfinite=True); optimizer.step(); losses.append(detail)
        metrics=evaluate(model,pairs,split['validation'],arrays,stats,a.device); value=score(metrics)
        if not math.isfinite(value): raise FloatingPointError('Nonfinite validation score')
        if value<best_score:
            best_state=cpu_state(model); best_score=value; best_epoch=epoch; stale=0
            save_model(out/'best.pt',model,stats,dict(stage='student',epoch=epoch))
        else: stale+=1
        elapsed=elapsed_before+time.monotonic()-started
        records.append(dict(epoch=epoch,elapsed_s=elapsed,train=aggregate(losses),validation=metrics['summary'],by_task=metrics['by_task'],selection_score=value))
        write_json(out/'history.json',dict(completed=False,selected_epoch=best_epoch,records=records))
        atomic_torch_save(out/'last.pt',dict(epoch=epoch,state_dict=cpu_state(model),optimizer=optimizer.state_dict(),records=records,best_score=best_score,best_state=best_state,best_epoch=best_epoch,stale=stale,elapsed_s=elapsed,rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []))
        emit(out,f"epoch={epoch} score={value:.6f} future_N={metrics['summary']['forecast_score_n']:.5f} reconstruction_nrmse={metrics['summary']['reconstruction_nrmse']:.5f} best_epoch={best_epoch} elapsed={elapsed/60:.1f}min")
    model.load_state_dict(best_state)
    if digest(model,frozen_only=True)!=frozen: raise RuntimeError('Frozen teacher weights changed')
    metrics=evaluate(model,pairs,split['validation'],arrays,stats,a.device,out/'predictions/validation')
    write_json(out/'metrics.json',metrics)
    summary=dict(completed=True,selected_epoch=best_epoch,selection_score=best_score,initial_selection_score=initial_score,records=records)
    write_json(out/'history.json',summary)
    checkpoint=out/'encoder_seed42.pt'; save_model(checkpoint,model,stats,dict(stage='student',epoch=best_epoch),motion_ready=False)
    manifest=dict(artifact_format=FORMAT,layout=LAYOUT,checkpoint=checkpoint.name,encoder_sha256=sha256(checkpoint),
                  stage='student',ready_domains=['real','sim'],history=HISTORY,past=PAST,future=FUTURE,policy_history=10,
                  motion_ready=False,selected_seed=split['seed'],tasks=sorted({x.task for x in native.values()}),
                  paired_episodes=len(pairs),parent_teacher_sha256=parent_manifest['encoder_sha256'],
                  force_contract_by_domain={'real':'original ee_force_base XYZ N','sim':'incoming link7 base-axis XYZ N; legacy peg aligned sign undone'},
                  sim_sign_calibration='new native incoming sign unverified by known-load probes; retained as recorded; separate Sim encoder trained',
                  status='source_mapped_single_holdout',actual_sim_state_verified=False,
                  quality_gate=dict(frozen_real_verified=True,improved_over_initialized_student=best_score<initial_score,
                                    future_prediction_improved=metrics['summary']['forecast_score_n']<initial['summary']['forecast_score_n']))
    write_json(out/'manifest.json',manifest); emit(out,f'Complete: {out}/manifest.json')


if __name__=='__main__': main()
