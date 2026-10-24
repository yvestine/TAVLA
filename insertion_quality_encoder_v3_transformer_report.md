# Insertion Quality V3 Transformer Audit

## Setup

- History window: `32` frames.
- Force input: existing 11D bounded force representation derived from Fx/Fy/Fz; Sim Tx/Ty/Tz excluded because the current field is a compatibility zero.
- Motion input for B/C: EE velocity, EE displacement from free-space reference, insertion depth.
- Labels: current reviewed `misaligned_contact`, `aligned_progress`, `terminal`; uncertain/free endpoints are masked. A long history is allowed to cross a phase boundary because the label describes the window end state.
- No TAVLA policy code was modified.

## Results

### D_transformer_pose_only

- Pooled Macro-F1: `0.757`
- Misaligned F1: `0.852`
- Aligned F1: `0.663`
- Accuracy: `0.794`
- AUROC: `0.849`
- Fold Macro-F1 mean: `0.721`
- Domain-head accuracy: `nan`

## Interpretation

The primary comparison is whether adding motion improves aligned-progress F1 over force-only Transformer, and whether C lowers domain-head accuracy without losing quality F1. The domain-head result is diagnostic; it is not sufficient by itself to justify integration.

The current V2 pooled baseline was Macro-F1 0.690, aligned F1 0.550, and latent domain accuracy 0.892. This V3 report must be compared against those values and against the handcrafted Logistic baseline 0.736.
