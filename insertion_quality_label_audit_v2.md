# Insertion Quality Label Audit V2

## 结论

已逐条复核 40 条轨迹的 Real/Sim front、wrist 视频，并与 Real/Sim 三维力和 EE pose 同步检查。标签是人工多证据审阅结果，不是由固定力阈值自动生成。

标签语义：`misaligned_contact` 表示已经接触但插入进展停滞/反弹且仍在纠偏；`aligned_progress` 表示最后一次纠偏后持续向孔内推进；`terminal` 与正确插入分开；无法证明质量的边界统一标为 `uncertain` 并在训练时 mask。

- 总轨迹：40
- 总帧数：7188
- 包含明确 misaligned 片段的轨迹：24
- 包含明确 aligned-progress 片段的轨迹：36
- 同时包含两种质量标签且可做轨迹内对照的轨迹：23
- 没有可信 aligned-progress、不可作为正类的轨迹：traj_12、traj_18、traj_30、traj_39

## 帧数分布

| label | frames | fraction | trajectories/segments |
|---|---:|---:|---:|
| free | 3137 | 43.6% | 40 |
| uncertain | 1121 | 15.6% | 41 |
| misaligned_contact | 1687 | 23.5% | 24 |
| aligned_progress | 984 | 13.7% | 36 |
| terminal | 259 | 3.6% | 39 |

按 10 帧历史且窗口不能跨阶段边界计算，可得到 `misaligned_contact=1471`、`aligned_progress=664` 个候选窗口；23 条轨迹同时含有两类可用窗口。训练时仍需按 trajectory/label 配平采样，不能直接按帧全量采样。

## 每条轨迹的最终边界

| trajectory | reviewed phases (inclusive frames) | video sheet | labeled signal |
|---|---|---|---|
| traj_0 | free 0-95 (high); uncertain 96-150 (high); misaligned_contact 151-220 (high); aligned_progress 221-255 (high); terminal 256-266 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_0_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_0_labeled.png) |
| traj_1 | free 0-96 (high); uncertain 97-112 (medium); misaligned_contact 113-128 (medium); aligned_progress 129-178 (medium); terminal 179-187 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_1_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_1_labeled.png) |
| traj_2 | free 0-103 (high); uncertain 104-145 (medium); aligned_progress 146-176 (medium); terminal 177-185 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_2_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_2_labeled.png) |
| traj_3 | free 0-95 (high); uncertain 96-155 (medium); aligned_progress 156-160 (medium); terminal 161-166 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_3_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_3_labeled.png) |
| traj_4 | free 0-114 (high); uncertain 115-143 (medium); aligned_progress 144-169 (medium); terminal 170-177 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_4_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_4_labeled.png) |
| traj_5 | free 0-105 (high); uncertain 106-133 (medium); aligned_progress 134-160 (medium); terminal 161-171 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_5_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_5_labeled.png) |
| traj_6 | free 0-100 (high); uncertain 101-117 (medium); misaligned_contact 118-140 (medium); aligned_progress 141-167 (medium); terminal 168-176 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_6_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_6_labeled.png) |
| traj_7 | free 0-109 (high); uncertain 110-119 (medium); misaligned_contact 120-229 (high); aligned_progress 230-244 (high); terminal 245-252 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_7_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_7_labeled.png) |
| traj_8 | free 0-88 (high); uncertain 89-109 (medium); misaligned_contact 110-129 (medium); aligned_progress 130-175 (medium); terminal 176-183 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_8_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_8_labeled.png) |
| traj_9 | free 0-93 (high); uncertain 94-134 (medium); misaligned_contact 135-314 (high); aligned_progress 315-359 (high); terminal 360-367 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_9_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_9_labeled.png) |
| traj_10 | free 0-92 (high); uncertain 93-135 (medium); misaligned_contact 136-285 (high); aligned_progress 286-313 (high); terminal 314-319 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_10_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_10_labeled.png) |
| traj_11 | free 0-103 (high); uncertain 104-120 (medium); misaligned_contact 121-159 (medium); aligned_progress 160-178 (medium); terminal 179-184 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_11_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_11_labeled.png) |
| traj_12 | free 0-91 (high); uncertain 92-109 (medium); misaligned_contact 110-224 (high); uncertain 225-255 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_12_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_12_labeled.png) |
| traj_13 | free 0-97 (high); uncertain 98-134 (medium); misaligned_contact 135-259 (high); aligned_progress 260-282 (high); terminal 283-290 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_13_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_13_labeled.png) |
| traj_14 | free 0-68 (high); uncertain 69-119 (medium); aligned_progress 120-145 (medium); terminal 146-152 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_14_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_14_labeled.png) |
| traj_15 | free 0-73 (high); uncertain 74-109 (medium); aligned_progress 110-127 (medium); terminal 128-132 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_15_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_15_labeled.png) |
| traj_16 | free 0-70 (high); uncertain 71-91 (medium); misaligned_contact 92-207 (high); aligned_progress 208-224 (high); terminal 225-229 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_16_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_16_labeled.png) |
| traj_17 | free 0-94 (high); uncertain 95-109 (medium); misaligned_contact 110-238 (high); aligned_progress 239-258 (high); terminal 259-264 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_17_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_17_labeled.png) |
| traj_18 | free 0-66 (high); uncertain 67-129 (medium); terminal 130-137 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_18_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_18_labeled.png) |
| traj_19 | free 0-50 (high); uncertain 51-57 (medium); aligned_progress 58-139 (medium); terminal 140-147 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_19_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_19_labeled.png) |
| traj_20 | free 0-107 (high); uncertain 108-128 (medium); misaligned_contact 129-219 (high); aligned_progress 220-229 (medium); terminal 230-236 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_20_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_20_labeled.png) |
| traj_21 | free 0-99 (high); uncertain 100-119 (medium); misaligned_contact 120-153 (medium); aligned_progress 154-190 (medium); terminal 191-196 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_21_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_21_labeled.png) |
| traj_22 | free 0-66 (high); uncertain 67-92 (medium); misaligned_contact 93-125 (high); aligned_progress 126-162 (high); terminal 163-168 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_22_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_22_labeled.png) |
| traj_23 | free 0-63 (high); uncertain 64-89 (medium); misaligned_contact 90-139 (high); aligned_progress 140-160 (high); terminal 161-166 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_23_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_23_labeled.png) |
| traj_24 | free 0-66 (high); uncertain 67-84 (medium); misaligned_contact 85-150 (high); aligned_progress 151-166 (medium); terminal 167-172 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_24_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_24_labeled.png) |
| traj_25 | free 0-77 (high); uncertain 78-113 (medium); misaligned_contact 114-124 (medium); aligned_progress 125-188 (medium); terminal 189-192 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_25_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_25_labeled.png) |
| traj_26 | free 0-62 (high); uncertain 63-67 (medium); aligned_progress 68-96 (medium); terminal 97-102 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_26_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_26_labeled.png) |
| traj_27 | free 0-44 (high); uncertain 45-77 (medium); misaligned_contact 78-125 (high); aligned_progress 126-143 (medium); terminal 144-149 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_27_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_27_labeled.png) |
| traj_28 | free 0-62 (high); uncertain 63-87 (medium); misaligned_contact 88-137 (medium); aligned_progress 138-157 (medium); terminal 158-163 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_28_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_28_labeled.png) |
| traj_29 | free 0-57 (high); uncertain 58-83 (medium); misaligned_contact 84-139 (high); aligned_progress 140-161 (high); terminal 162-167 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_29_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_29_labeled.png) |
| traj_30 | free 0-53 (high); uncertain 54-97 (medium); terminal 98-104 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_30_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_30_labeled.png) |
| traj_31 | free 0-51 (high); uncertain 52-73 (medium); aligned_progress 74-95 (medium); terminal 96-101 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_31_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_31_labeled.png) |
| traj_32 | free 0-81 (high); uncertain 82-98 (medium); misaligned_contact 99-124 (medium); aligned_progress 125-149 (medium); terminal 150-152 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_32_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_32_labeled.png) |
| traj_33 | free 0-65 (high); uncertain 66-79 (medium); misaligned_contact 80-107 (medium); aligned_progress 108-121 (medium); terminal 122-127 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_33_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_33_labeled.png) |
| traj_34 | free 0-70 (high); uncertain 71-94 (medium); misaligned_contact 95-195 (high); aligned_progress 196-207 (medium); terminal 208-213 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_34_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_34_labeled.png) |
| traj_35 | free 0-54 (high); uncertain 55-77 (medium); aligned_progress 78-110 (medium); terminal 111-116 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_35_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_35_labeled.png) |
| traj_36 | free 0-57 (high); uncertain 58-72 (medium); aligned_progress 73-94 (medium); terminal 95-99 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_36_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_36_labeled.png) |
| traj_37 | free 0-52 (high); uncertain 53-69 (medium); aligned_progress 70-93 (medium); terminal 94-98 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_37_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_37_labeled.png) |
| traj_38 | free 0-66 (high); uncertain 67-91 (medium); aligned_progress 92-109 (medium); terminal 110-114 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_38_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_38_labeled.png) |
| traj_39 | free 0-43 (high); uncertain 44-69 (medium); terminal 70-74 (high) | [videos](eval_outputs/insertion_quality_label_review_v2/traj_39_videos.jpg) | [signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_39_labeled.png) |

## 使用限制

- 这是一版严格保守标签：`uncertain` 不可转成 aligned 或 misaligned。
- Real/Sim 视频用于共同复核运动阶段，但两域 force timing 不一致，因此标签不等于同帧力学状态完全一致。
- `medium` 表示片段语义可信但边界可能有约 3-5 帧误差；训练窗口若跨越边界应继续 mask。
- `traj_12` 是重要的错误插入/撤回负例；不能因为轨迹结束就自动添加 aligned-progress。
- 下一步应先用这些标签重新统计每折可用正负窗口与类别平衡，再决定是否训练 Encoder。
