"""Check that policy comparisons use full-size single-GPU configs and fresh mixture epochs."""
import unittest
import numpy as np
from openpi.training.config import get_config
from openpi.training.data_loader import TransformedDataset, WeightedMixtureDataset


class Records:
    def __init__(self,n,domain): self.n,self.domain=n,domain
    def __len__(self): return self.n
    def __getitem__(self,i): return {'id':int(i),'domain':self.domain}


class Host9Setup(unittest.TestCase):
    def test_larger_simulation_source_is_resampled_each_epoch(self):
        ds=WeightedMixtureDataset([Records(100,'real'),Records(200,'sim')],[.5,.5],seed=42,sampling_mode='epoch_without_replacement')
        wrapped=TransformedDataset(ds,[])
        wrapped.set_epoch(0); first=[wrapped[i] for i in range(len(wrapped))]
        wrapped.set_epoch(1); second=[wrapped[i] for i in range(len(wrapped))]
        real=lambda rows:{x['id'] for x in rows if x['domain']=='real'}
        sim=lambda rows:{x['id'] for x in rows if x['domain']=='sim'}
        self.assertEqual(real(first),set(range(100)))
        self.assertEqual(real(second),set(range(100)))
        self.assertEqual(len(sim(first)),100)
        self.assertNotEqual(sim(first),sim(second))
        wrapped.set_epoch(0)
        self.assertEqual(first,[wrapped[i] for i in range(len(wrapped))])

    def test_four_models_have_comparable_task_inputs_and_single_gpu(self):
        for task in ('dual_peg','usb'):
            raw=get_config(f'pi0_lora_host9_{task}_raw6_cotrain')
            enc=get_config(f'pi0_lora_host9_{task}_encoder_cotrain')
            for cfg in (raw,enc):
                self.assertEqual(cfg.batch_size,8); self.assertEqual(cfg.fsdp_devices,1)
                self.assertEqual(cfg.data.dataset_sampling_weights,(.5,.5))
                self.assertEqual(cfg.data.dataset_sampling_mode,'epoch_without_replacement')
            self.assertEqual(raw.model.effort_dim,6); self.assertEqual(enc.model.effort_dim,32)
            self.assertEqual(tuple(enc.data.effort_history),tuple(range(-9,1)))
            self.assertEqual(raw.num_train_steps,enc.num_train_steps)


if __name__=='__main__': unittest.main()
