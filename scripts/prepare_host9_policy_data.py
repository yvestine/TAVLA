"""Convert manifest-approved Host9 demonstrations to task-specific LeRobot data."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import cv2
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import serialize_dict, write_json

from force_gap_forecast_v1.common import sha256
from force_predictive_encoder_v3 import LAYOUT
from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
from openpi.shared.image_tools import resize_with_pad

TASKS={'dual_peg':'dual_peg_insert','usb':'usb_insert','plate_sliding':'plate_slide'}
STATE_NAMES=tuple(f'joint_{i}' for i in range(7))+('gripper',)
WRENCH_NAMES=('Fx','Fy','Fz','Tx','Ty','Tz')


def read_episode(path,domain):
    with h5py.File(path,'r') as h:
        if domain=='real':
            state=np.concatenate([h['obs/state/joint_pos'][:],h['obs/state/gripper_pos'][:]],-1)
            action=np.concatenate([h['action/actual/arm'][:],h['action/actual/gripper'][:]],-1)
            effort=h['obs/state/ee_wrench_base'][:]; clock=h['timestamps'][:]
        else:
            state=h['observations/qpos'][:]; action=h['action'][:]
            effort=h['observations/effort'][:]; clock=h['timestamp'][:]
            if not np.array_equal(h['observations/video_frame_index'][:],np.arange(len(clock))):
                raise ValueError(f'{path}: observation/video index mapping differs')
            if not np.allclose(h['observations/wrench_base'][:],effort,atol=1e-6):
                raise ValueError(f'{path}: effort differs from declared native wrench')
            if 'absolute' not in str(h.attrs.get('action_semantics','')):
                raise ValueError(f'{path}: action must be an absolute joint/gripper target')
    n=len(clock)
    if state.shape!=(n,8) or action.shape!=(n,8) or effort.shape!=(n,6): raise ValueError(f'{path}: incompatible shape')
    if not all(np.isfinite(v).all() for v in (state,action,effort,clock)) or np.any(np.diff(clock)<=0):
        raise ValueError(f'{path}: invalid values or timestamps')
    return state.astype(np.float32),action.astype(np.float32),effort.astype(np.float32),clock


def video_info(path,n):
    cap=cv2.VideoCapture(str(path)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok=cap.isOpened(); cap.release()
    if not ok or count!=n: raise ValueError(f'{path}: video frames {count}, expected {n}')


def read_video(path,n):
    """Use exactly the model's letterbox resize, once, before caching images."""
    cap=cv2.VideoCapture(str(path)); result=[]; block=[]
    def flush():
        if block:
            images=np.asarray(block,np.uint8)
            small=np.asarray(resize_with_pad(jnp.asarray(images),224,224))
            result.extend(np.transpose(small,(0,3,1,2)))
            block.clear()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            block.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if len(block)==16: flush()
        flush()
    finally: cap.release()
    if len(result)!=n: raise ValueError(f'{path}: decoded {len(result)} frames, expected {n}')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('host9_rollouts_20261026_105valid'))
    p.add_argument('--task',choices=TASKS,required=True); p.add_argument('--kind',choices=('raw6','encoder'),required=True)
    p.add_argument('--domain',choices=('real','sim'),required=True)
    p.add_argument('--version',default='20261026_v1'); p.add_argument('--repo-id')
    p.add_argument('--encoder-manifest',type=Path,default=Path('eval_outputs/force_predictive_v3_multitask/host9_student_rollouts_v1/manifest.json'))
    p.add_argument('--audit-only',action='store_true'); p.add_argument('--max-trajectories',type=int)
    a=p.parse_args(argv); root=a.bundle.resolve()
    entries=[e for e in json.loads((root/'training_manifest.json').read_text())['episodes']
             if e['task']==TASKS[a.task] and e['state']=='eligible']
    entries=sorted(entries,key=lambda e:int(e['trajectory'].split('_')[-1]))
    if a.max_trajectories: entries=entries[:a.max_trajectories]
    encoder=None; encoder_sha=None
    if a.kind=='encoder':
        em=json.loads(a.encoder_manifest.read_text())
        gate=em.get('quality_gate',{})
        if em.get('ready_domains')!=['real','sim'] or not all(gate.get(k) for k in ('frozen_real_verified','improved_over_initialized_student','future_prediction_improved')):
            raise ValueError('Encoder must finish both domains and pass its training checks first')
        encoder_sha=em['encoder_sha256']
        if not a.audit_only:
            encoder=ForcePredictiveEncoder(a.encoder_manifest,a.domain,device='cpu',threads=2)
            encoder.model.requires_grad_(False)
    records=[]; prompts=set(); frames=0
    for e in entries:
        path=root/(e['source'] if a.domain=='real' else e['data_h5'])
        state,action,effort,clock=read_episode(path,a.domain)
        with h5py.File(root/e['data_h5'],'r') as h: prompt=str(h.attrs['task'])
        prompts.add(prompt)
        for file in ('front_camera.mp4','wrist_camera.mp4'): video_info(path.parent/file,len(clock))
        records.append(dict(trajectory=e['trajectory'],path=str(path),sha256=sha256(path),frames=len(clock),
                            front_sha256=sha256(path.parent/'front_camera.mp4'),wrist_sha256=sha256(path.parent/'wrist_camera.mp4')))
        frames+=len(clock)
    if len(prompts)!=1 or not records: raise ValueError('Expected one task prompt and nonempty eligible episodes')
    prompt=prompts.pop()
    contract=dict(format='host9_policy_data_v1',task=a.task,kind=a.kind,domain=a.domain,
                  bundle_manifest_sha256=sha256(root/'training_manifest.json'),encoder_sha256=encoder_sha,
                  sources=records,prompt=prompt,fps=10,image_shape=[3,224,224],
                  action_source='domain_native_actual_absolute',sim_force='native exported incoming wrench; no sign/scale fitting',
                  gap_policy='split episodes at timestamp gaps >0.5s',whitelist='eligible rollout IDs used for both Real and Sim')
    repo=a.repo_id or f'local/tavla_host9_{a.version}_{a.task}_{a.kind}_{a.domain}'
    target=LEROBOT_HOME/repo; marker=target/'meta/host9_source.json'
    print(f'{repo}: trajectories={len(records)} frames={frames} prompt={prompt!r}',flush=True)
    if a.audit_only: return
    # Parallel preparation and the pipeline may request the same dataset.
    # Keep the lock outside the dataset so interrupted-output renames are safe.
    target.parent.mkdir(parents=True,exist_ok=True)
    conversion_lock=(target.parent/f'.{target.name}.conversion.lock').open('a+')
    fcntl.flock(conversion_lock,fcntl.LOCK_EX)
    if marker.is_file():
        saved=json.loads(marker.read_text())
        if saved.get('contract')!=contract: raise ValueError(f'{target}: source or encoder changed; select a new version')
        if not (target/'meta/stats.json').is_file(): raise ValueError(f'{target}: completed dataset lacks statistics')
        print(f'Reusing verified dataset {target}',flush=True); return
    if target.exists():
        request=target/'host9_conversion_request.json'
        if not request.is_file() or json.loads(request.read_text())!=contract:
            raise FileExistsError(f'{target}: existing data without matching conversion request')
        backup=target.with_name(target.name+f'.partial_{time.time_ns()}'); target.rename(backup)
        print(f'Preserved interrupted conversion at {backup}',flush=True)
    names=WRENCH_NAMES if a.kind=='raw6' else tuple(LAYOUT); dim=len(names)
    image_feature=dict(dtype='image',shape=(3,224,224),names=['channels','height','width'])
    features={'observation.state':dict(dtype='float32',shape=(8,),names=[STATE_NAMES]),
              'action':dict(dtype='float32',shape=(8,),names=[STATE_NAMES]),
              'observation.effort':dict(dtype='float32',shape=(dim,),names=[names]),
              **{f'observation.images.{k}':image_feature for k in ('cam_high','cam_left_wrist','cam_right_wrist')}}
    ds=LeRobotDataset.create(repo_id=repo,fps=10,robot_type='single_arm',features=features,
                            use_videos=False,image_writer_processes=0,image_writer_threads=0)
    (target/'host9_conversion_request.json').write_text(json.dumps(contract,indent=2))
    segments=[]
    for record in records:
        path=Path(record['path']); state,action,effort,clock=read_episode(path,a.domain)
        if encoder is not None: effort=encoder.batch(effort[:,:3],clock).astype(np.float32)
        front=read_video(path.parent/'front_camera.mp4',len(clock)); wrist=read_video(path.parent/'wrist_camera.mp4',len(clock))
        boundaries=np.r_[0,np.flatnonzero(np.diff(clock)>.5)+1,len(clock)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            for i in range(int(lo),int(hi)):
                ds.add_frame({'observation.state':state[i],'action':action[i],'observation.effort':effort[i],
                              'observation.images.cam_high':front[i],'observation.images.cam_left_wrist':wrist[i],
                              'observation.images.cam_right_wrist':wrist[i]})
            ds.save_episode(task=prompt)
            segments.append(dict(trajectory=record['trajectory'],start=int(lo),end=int(hi)))
        print(f"{record['trajectory']}: wrote {len(clock)} frames",flush=True)
    ds.consolidate(run_compute_stats=False)
    ds.meta.stats=compute_stats(ds,batch_size=8,num_workers=0)
    write_json(serialize_dict(ds.meta.stats),target/'meta/stats.json')
    if ds.meta.total_frames!=frames: raise ValueError(f'Output frame count {ds.meta.total_frames} != {frames}')
    write_json(dict(completed=True,contract=contract,segments=segments,total_frames=frames,total_episodes=len(segments)),marker)
    print(f'Complete: {target}',flush=True)


if __name__=='__main__': main()
