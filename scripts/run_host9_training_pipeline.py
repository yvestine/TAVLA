"""Finish the force encoder, prepare paired experiments, then run one policy per GPU."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used',
                                    '--format=csv,noheader,nounits'],text=True)
    return {int(row[0]):dict(uuid=row[1].strip(),free_mib=int(row[2]),used_mib=int(row[3]))
            for row in csv.reader(output.splitlines())}


def logged(command,env,path):
    print('RUN',shlex.join(map(str,command)),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as log:
        process=subprocess.Popen(list(map(str,command)),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line); print(line,end='',flush=True)
        if process.wait(): raise RuntimeError(f'Command failed; see {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('configs/host9_training_plan_20261026.json'))
    p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    project=Path(__file__).resolve().parents[1]; os.chdir(project)
    plan=json.loads(a.plan.read_text()); version=plan['version']; py=project/'.venv/bin/python'
    run=project/'eval_outputs/host9_training_pipeline'/version; run.mkdir(parents=True,exist_ok=True)
    if not a.check_only:
        lock=(run/'pipeline.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    state=dict(stage='preflight',plan=plan,jobs=[])
    def status():
        state['updated_unix_s']=time.time(); temp=run/'status.json.tmp'; temp.write_text(json.dumps(state,indent=2)); temp.replace(run/'status.json')
    status()
    env={**os.environ,'PYTHONPATH':f'{project}/scripts:{project}/src:{project}/packages/openpi-client/src',
         'OPENPI_DATA_HOME':str(project.parent/'.cache/openpi'),
         'LEROBOT_HOME':str(project.parent/'.cache/huggingface/lerobot'),
         'HOST9_DATASET_VERSION':version,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONUNBUFFERED':'1',
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    cpu={**env,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    encoder=project/plan['encoder_run']; em=encoder/'manifest.json'
    e=plan['encoder']; inventory=gpu_inventory()
    if a.check_only:
        print(json.dumps(dict(plan=plan,gpus=inventory,python_exists=py.is_file(),base_params_exists=(project.parent/'.cache/openpi/openpi-assets/checkpoints/pi0_base/params').is_dir()),indent=2)); return
    state['stage']='encoder'; status()
    ecmd=[py,'-m','force_predictive_encoder_v3.rollout_student','--teacher-run',project/plan['teacher_run'],
          '--output-dir',encoder,'--batch-size',e['batch_size'],'--max-epochs',e['max_epochs'],
          '--patience',e['patience'],'--align-weight',e['align_weight']]
    if (encoder/'recipe.json').exists(): ecmd.append('--resume')
    logged(ecmd,{**env,'CUDA_VISIBLE_DEVICES':inventory[e['gpu']]['uuid']},project/'logs/host9_encoder_student.log')
    manifest=json.loads(em.read_text())
    gate=manifest.get('quality_gate',{})
    if manifest.get('ready_domains')!=['real','sim'] or not all(gate.get(k) for k in ('frozen_real_verified','improved_over_initialized_student','future_prediction_improved')):
        raise RuntimeError('Encoder quality checks did not pass; policy jobs were not started')
    if hashlib.sha256((encoder/manifest['checkpoint']).read_bytes()).hexdigest()!=manifest['encoder_sha256']:
        raise RuntimeError('Encoder SHA changed')
    def advance_jobs():
        inventory=gpu_inventory()
        for job in state['jobs']:
            if job['status']=='queued':
                gpu=inventory[job['gpu']]
                if gpu['free_mib']<plan['minimum_free_gpu_mib']:
                    job['waiting_for_free_mib']=gpu['free_mib']; continue
                if subprocess.run(['tmux','has-session','-t',job['tmux']],stderr=subprocess.DEVNULL).returncode==0:
                    raise RuntimeError(f"tmux {job['tmux']} already exists; inspect it before resuming orchestration")
                command=[str(py),'scripts/train.py',job['config'],'--exp-name',version,'--batch-size',str(plan['policy']['batch_size']),
                         '--fsdp-devices','1','--num-workers','0','--num-train-steps',str(plan['policy']['num_train_steps']),
                         '--save-interval','5000','--no-wandb-enabled']
                cp=Path(job['checkpoint'])
                if cp.exists():
                    if not any(x.name.isdigit() for x in cp.iterdir()): raise RuntimeError(f'{cp}: existing run without a checkpoint')
                    command.append('--resume')
                launch_env={k:env[k] for k in ('PYTHONPATH','OPENPI_DATA_HOME','LEROBOT_HOME','HOST9_DATASET_VERSION',
                                               'XLA_PYTHON_CLIENT_PREALLOCATE','PYTHONUNBUFFERED','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}
                launch_env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],JAX_PLATFORMS='cuda')
                shell=shlex.join(['env',*[f'{k}={v}' for k,v in launch_env.items()],*command])+f" 2>&1 | tee -a {shlex.quote(job['log'])}"
                subprocess.run(['tmux','new-session','-d','-s',job['tmux'],'-c',str(project),
                                'bash -o pipefail -c '+shlex.quote(shell)],check=True)
                subprocess.run(['tmux','set-option','-t',job['tmux'],'remain-on-exit','on'],check=True)
                job.update(status='running',started_unix_s=time.time(),gpu_uuid=gpu['uuid'])
                print(f"STARTED {job['config']} GPU{job['gpu']} batch={plan['policy']['batch_size']}",flush=True)
            elif job['status']=='running':
                pane=subprocess.check_output(['tmux','display-message','-p','-t',job['tmux'],'#{pane_dead} #{pane_dead_status}'],text=True).strip().split()
                if pane and pane[0]=='1':
                    final=Path(job['checkpoint'])/str(plan['policy']['num_train_steps']-1)
                    job['status']='complete' if len(pane)>1 and pane[1]=='0' and final.is_dir() else 'failed'
                    print(job['config'],job['status'],flush=True)

    state.update(stage='prepare_data',encoder_sha256=manifest['encoder_sha256']); status()
    for job in plan['jobs']:
        state['stage']='prepare_data'; status()
        for domain in ('real','sim'):
            logged([py,'scripts/prepare_host9_policy_data.py','--task',job['task'],'--kind',job['kind'],
                    '--domain',domain,'--version',version,'--encoder-manifest',em],cpu,
                   project/f"logs/host9_prepare_{job['task']}_{job['kind']}_{domain}.log")
        state['stage']='norm_stats'; status()
        name=f"pi0_lora_host9_{job['task']}_{job['kind']}_cotrain"
        logged([py,'scripts/compute_norm_stats.py','--config-name',name,'--batch-size','8','--num-workers','0'],cpu,
               project/f'logs/{name}_stats.log')
        state['jobs'].append({**job,'config':name,'status':'queued','log':str(project/f"logs/{job['tmux']}.log"),
                              'checkpoint':str(project/f'checkpoints/{name}/{version}')})
        advance_jobs(); status()
    # Raw and encoded versions must refer to the same episodes, labels and images.
    for task in ('dual_peg','usb'):
        for domain in ('real','sim'):
            docs=[]
            for kind in ('raw6','encoder'):
                meta=Path(env['LEROBOT_HOME'])/f'local/tavla_host9_{version}_{task}_{kind}_{domain}'/'meta/host9_source.json'
                docs.append(json.loads(meta.read_text()))
            if docs[0]['contract']['sources']!=docs[1]['contract']['sources'] or docs[0]['segments']!=docs[1]['segments']:
                raise RuntimeError(f'{task}/{domain}: raw and encoded data differ')
            if docs[1]['contract']['encoder_sha256']!=manifest['encoder_sha256']:
                raise RuntimeError('Encoded dataset uses different weights')
    state['stage']='policies'; status()
    while any(j['status'] not in ('complete','failed') for j in state['jobs']):
        advance_jobs()
        status(); time.sleep(30)
    state['stage']='complete' if all(j['status']=='complete' for j in state['jobs']) else 'failed'; status()
    if state['stage']=='failed': raise RuntimeError('At least one policy job failed; inspect its log')


if __name__=='__main__': main()
