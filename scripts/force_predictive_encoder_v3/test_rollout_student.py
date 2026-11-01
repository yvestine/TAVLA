"""Regression checks for time-warped labels, gaps, and the frozen teacher."""
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from force_gap_forecast_v1.data import segment_limits
from .model import PredictiveForce
from .multitask_data import ForceEpisode, prepare_arrays
from .rollout_student import MappedBlocks, interpolate_reference, supervision
from .losses import objective
from .train import digest


class RecordedSupervision(unittest.TestCase):
    def episode(self, time):
        starts,ends=segment_limits(time)
        force=np.repeat(np.arange(len(time),dtype=np.float32)[:,None],3,axis=1)
        return ForceEpisode('task',0,force,time,starts,ends,Path('synthetic.h5'),'synthetic')

    def test_future_is_in_original_real_frames(self):
        ep=self.episode(np.arange(20)*.1)
        mapped=supervision(ep,np.array([0.,.5,1.,1.,3.5]))
        self.assertAlmostEqual(mapped['future'][1,0,0],1.5)
        self.assertAlmostEqual(mapped['future'][1,4,0],5.5)
        # A paused source index does not turn future labels into a constant.
        np.testing.assert_array_equal(mapped['future'][2],mapped['future'][3])

    def test_no_interpolation_or_future_target_crosses_recording_gap(self):
        t=np.arange(20)*.1; t[3:]+=1
        ep=self.episode(t); mapped=supervision(ep,np.array([1.,2.,2.5,3.,19.]))
        self.assertFalse(mapped['valid'][2])
        self.assertTrue(mapped['future_mask'][0,0])
        self.assertFalse(mapped['future_mask'][0,1])
        self.assertFalse(mapped['future_mask'][-1].any())
        _,valid=interpolate_reference(ep.force,np.array([-1.,np.nan,20.]),ep.starts)
        self.assertFalse(valid.any())

    def test_student_update_keeps_real_branch_and_head_fixed(self):
        torch.set_num_threads(1); torch.manual_seed(5)
        ep=self.episode(np.arange(40)*.1)
        stats={d:dict(mean=[0.,0.,0.],scale=[10.,10.,10.]) for d in ('real','sim','target')}
        model=PredictiveForce(stats); model.initialize_student(); model.configure('student')
        arrays=prepare_arrays({'task/traj_0':ep},stats,'sim')
        pair=dict(ep=ep,index=np.arange(40),**supervision(ep,np.arange(40)))
        data=MappedBlocks({'task/traj_0':pair},['task/traj_0'],arrays,{'task/traj_0':np.zeros((40,32),np.float32)},1.)
        batch=next(iter(DataLoader(data,batch_size=2,sampler=[20,21])))
        x=batch['x']; b,k=x.shape[:2]; before=digest(model,frozen_only=True)
        student_before={n:v.clone() for n,v in model.encoders[1].state_dict().items()}
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-3)
        out={n:v.reshape(b,k,*v.shape[1:]) for n,v in model(x.reshape(b*k,*x.shape[2:]),1).items()}
        loss,_=objective(model,out,batch,0,.1,{'z':batch['teacher_z']})
        loss.backward(); opt.step()
        self.assertEqual(before,digest(model,frozen_only=True))
        self.assertTrue(any(not torch.equal(v,model.encoders[1].state_dict()[n]) for n,v in student_before.items()))


if __name__=='__main__': unittest.main()
