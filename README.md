<div align="center">
<h2>TA-VLA: Elucidating the Design Space of <br>
Torque-aware Vision-Language-Action Models</h2>

  **CoRL 2025**

**Zongzheng Zhang**<sup>*1</sup> · [**Haobo Xu**](https://hsu1023.github.io/)<sup>*2</sup> · **Zhuo Yang**<sup>*1</sup> ·<br>
**Chenghao Yue**<sup>1</sup> · **Zehao Lin**<sup>1</sup> · [**Huan-ang Gao**](https://c7w.tech/)<sup>1</sup>· [**Ziwei Wang**](https://ziweiwangthu.github.io/)<sup>3</sup> .  [**Hao Zhao**](https://sites.google.com/view/fromandto/)<sup>1,2</sup><br>

<sup>1</sup> Beijing Academy of Artificial Intelligence (BAAI), <br> 
<sup>2</sup> Institute for AI Industry Research (AIR), Tsinghua University, <br>
<sup>3</sup> Nanyang Technological University<br>
<sub>(* indicates equal contribution)</sub><br>

[**Project Page**](https://zzongzheng0918.github.io/Torque-Aware-VLA.github.io/) | [**arXiv**](https://arxiv.org/abs/2509.07962) | [**Code**](https://github.com/ZZongzheng0918/TA-VLA) 
</div>

## TA-VLA
This repository provides the implementation of **TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models** on [**openpi**](https://github.com/Physical-Intelligence/openpi).
It is branched from the original repository at commit `cd82848`.
Please refer to the original repository for environment setup and training details.
The following focuses only on the differences from the upstream repository.

## Torque Input Types
The file `src/openpi/shared/effort_type.py` defines the ways in which torque information is fed into openpi, covering all the experiments described in the paper.  
The types are:

- **NO**  
  No effort is used, but `TavlaInputs` still processes it for `norm_stats` computation.  
  Used for the baseline model.

- **STATE**  
  Inserts the current effort into the last `state[-14:]` so that it will be considered by the action expert.  
  Corresponds to the *DePre* method in Section 4.1.

- **LLM**  
  Projects the current effort into a token and passes it to the LLM along with image and language tokens.  
  Projector MLP: `Linear(in, 2*w) -> swish -> Linear(2*w, w)`.  
  Corresponds to the *Enc* method in Section 4.1.

- **LLM_HIS_C**  
  Concatenates current and historical effort, projects it into a token, and passes it to the LLM.  
  Corresponds to *Enc-1* in Section 4.2.

- **LLM_HIS_T**  
  Projects current and historical effort into tokens separately and passes them to the LLM.  
  Corresponds to *Enc-H* in Section 4.2.

- **EXPERT**  
  Projects effort into a token and passes it to the action expert (a component of the LLM) together with state and action tokens.  
  Corresponds to *DePost* in Section 4.1.

- **EXPERT_HIS_C**  
  Concatenates current and historical effort, projects it into a token, and passes it to the action expert.  
  Corresponds to *Dec-1* in Section 4.2.

- **EXPERT_HIS_T**  
  Projects current and historical effort into tokens separately and passes them to the action expert.  
  Corresponds to *Dec-H* in Section 4.2.

- **EXPERT_FUT**  
  Not an input type per se, but predicts future effort along with actions.  
  Corresponds to Sections 5 and 6 ($π_0$ + obj).

- **EXPERT_HIS_C_FUT**  
  Inputs concatenated historical effort to the action expert and outputs future effort.  
  Corresponds to Sections 5 and 6 ($π_0$ + obs + obj).

- **EXPERT_HIS_C_L_FUT**  
  Inputs concatenated historical effort as the last token and outputs future effort.  
  A positional variant of the previous type, tested without performance improvement.

Note: These torque-handling implementations have only been tested on $π_0$ and may not be compatible with $π_0$-FAST.

## Dataset
As in the original openpi implementation, we use datasets in the standard **lerobot** format.
The difference is that we expect an additional field `observation.effort` storing the per-frame joint torque, analogous to how `observation.state` stores per-frame joint angles.

## Training
Refer to the example configurations provided in `src/openpi/training/config.py`.
When using effort inputs, be sure to pass the corresponding `effort_history` parameter.

## This fork: single-arm wrench LoRA fine-tuning

This working tree additionally contains single-arm HDF5-to-LeRobot conversion, joint-effort and six-dimensional end-effector wrench inputs, LoRA fine-tuning configurations, simulation co-training utilities, wrench adaptation utilities, offline evaluation, and reset-aware policy serving.

For a clean-server installation and migration procedure, see [`docs/migration.md`](docs/migration.md). The short command-oriented workflow is also kept in [`run.md`](run.md).

Model weights, datasets, local caches, checkpoints, and evaluation outputs are intentionally excluded from Git. The public `pi0_base` weights are downloaded into the OpenPI cache by `scripts/download_base_checkpoint.py` or automatically on first use.

## Deployment
For data collection and model deployment, we use a modified version of the AgileX official example code.
In addition to reading torque values from the ROS topic, this version maintains a historical torque buffer for policies that require past torque information.

## Citation
If you find this project useful, feel free to cite our work!
<div style="display:flex;">
<div>

```bibtex
@article{zhang2025ta,
  title={TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models},
  author={Zhang, Zongzheng and Xu, Haobo and Yang, Zhuo and Yue, Chenghao and Lin, Zehao and Gao, Huan-ang and Wang, Ziwei and Zhao, Hao},
  journal={arXiv preprint arXiv:2509.07962},
  year={2025}
}
```


