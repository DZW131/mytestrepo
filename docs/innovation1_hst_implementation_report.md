# Innovation 1 HST Implementation Report

## 1. Milestone status

This branch implements and validates the specification through **Phase C / A1
progressive-only**. It deliberately stops before A2 transition MLPs, A3 latent
interaction, or any full dataset training.

Completed:

- Phase A: switchable `hfrm` / `hst` rectifier interface and upstream baseline
  equivalence guard;
- Phase B: hierarchy-specific semantic projectors into a shared 256-dimensional
  latent space;
- Phase C: deep-to-shallow correction-state propagation, stage-specific gates,
  unchanged SSHR Contextual Homogenization (CH), and zero-initialized residual
  scales;
- diagnostics, checkpoint/logging controls, unit tests, CUDA smoke tests, and
  efficiency profiling.

Not started by design:

- A2 stage-specific transition blocks and learnable `rho`;
- A3 MLP/attention hierarchy token interaction;
- RawFeatureCascade analysis control;
- BCSS or LUAD full training.

## 2. Mathematical mapping to code

For a 224×224 image, the actual SSHR hierarchy is:

| Symbol | SSHR tensor | Channels | Spatial size |
|---|---|---:|---:|
| \(F_1\) | `feat_56` / CAM56 feature | 256 | 56×56 |
| \(F_2\) | `feat_28_1` | 512 | 28×28 |
| \(F_3\) | `feat_28_2` | 1024 | 28×28 |
| \(F_D\) | `feat_deep` | 4096 | 28×28 |

Each hierarchy descriptor is implemented by
`HierarchySemanticProjector`:

```text
z_i = LayerNorm(Linear(GAP(F_i)))  # latent_dim = 256
```

The A1 correction path is:

```text
C_D = phi_D(z_D)
C_3 = C_D
C_2 = C_3
C_1 = C_2
```

`phi_D` is a learnable bias-free linear map initialized to identity. Each target
stage owns a separate semantic gate:

```text
w_i = sigmoid(W_i(C_i))
S_i = F_i * w_i
F_i_R = F_i + gamma_sem_i * S_i + gamma_ctx_i * CH_i(F_i)
```

Both residual scales are scalar parameters initialized to zero. Consequently,
the initial A1 rectified features are exactly the input hierarchy features. The
correction state, rather than a raw spatial feature map, is the only quantity
propagated down the hierarchy.

## 3. HFRM replacement boundary

Only the public HFRM semantic-guidance branch is replaceable. HST preserves:

- ResNet38 backbone and frozen-layer behavior;
- CAM56, CAM28_1, CAM28_2, and CAMdeep heads;
- the original depthwise `K=15` CH convolution, padding, no-bias setting,
  uniform `1/225` initialization, and zero-initialized context gamma;
- classification loss weights `0.10 / 0.15 / 0.25 / 0.50`;
- public PolyOptimizer, learning-rate schedule, augmentation, inference fusion,
  threshold, and official metric.

The default constructor remains `rectifier_type="hfrm"`. No HST parameters or
state-dict keys are created on that path.

## 4. Files and public APIs

| File | Responsibility |
|---|---|
| `network/hst/context.py` | Single CH factory/apply implementation shared by HFRM and HST |
| `network/hst/semantic_projector.py` | GAP → Linear → LayerNorm hierarchy projection |
| `network/hst/latent_interaction.py` | Exact identity interface for A1; rejects unvalidated mixers |
| `network/hst/hst_rectifier.py` | `HSTConfig` and progressive-only `HSTRectifier` |
| `network/resnet38_cls.py` | Backbone feature extraction, rectifier switch, unchanged CAM API, diagnostics |
| `train_sshr.py` | CLI switch, exact experiment manifest, validation-only selection and checkpoint records |
| `tests/test_hst_a1.py` | Baseline compatibility and A1 component/integration checks |
| `tools/smoke_hst_a1.py` | Dataset-free two-step training-path smoke test |
| `tools/profile_hst.py` | Reproducible parameters/FLOPs/runtime comparison |
| `tools/check_pretrained_load.py` | Pretrained conversion/load dry run with key audit |

Existing training callers still receive the original 10-tensor tuple. Analysis
code may call `Net.forward_with_diagnostics(x)` to additionally retrieve:

- base and rectified features;
- semantic descriptors;
- `C_D`, `C_3`, `C_2`, `C_1` correction states;
- stage semantic gates and semantic/context feature contributions;
- per-stage CAM logits.

## 5. Verification evidence

### 5.1 Baseline equivalence

For default `rectifier=hfrm`, comparison against the verbatim upstream forward
path at commit `7346cc5` shows:

- total parameters: `112,709,714` before and after;
- state-dict keys: 260 before and after;
- ordered state-key SHA256:
  `23038075b660d3f97ada855a9c138cda6f82711902214c2aa70e6a394e45b796`;
- fixed-seed/fixed-input outputs: exact `torch.equal` for all 10 returned tensors.

This is stronger than tolerance-based closeness for the tested deterministic
forward.

### 5.2 Automated tests

Command:

```bash
python -m unittest discover -s tests -v
```

Result on Windows CPU and the 5090 server: **8/8 passed**.

| Check | Result |
|---|---|
| Upstream HFRM state layout and output equivalence | Pass |
| HLI identity returns the same tensor | Pass |
| Zero-gamma A1 returns unchanged features | Pass |
| Correction-state progression and tensor shapes | Pass |
| Original HFRM CH vs HST CH | Exact equality |
| Finite outputs and two-step gradient connectivity | Pass |
| Every trainable parameter appears in one optimizer group | Pass |
| Backbone/BN freeze does not freeze HST | Pass |

The A1 target projectors for `z_3/z_2/z_1` are intentionally dormant because
A1 defines `C_i=C_parent`; A2 will consume them. The deep projector and all
stage gates become active after the zero-initialized semantic gamma takes its
first optimizer update.

### 5.3 CUDA two-step smoke

Environment:

- GPU: NVIDIA GeForce RTX 5090 D v2 (24 GB);
- Python 3.10.20;
- PyTorch 2.11.0+cu128;
- batch 2, 224×224 random input, frozen baseline classification loss.

Result:

- step-1 loss `0.6873157`, step-2 loss `0.6855530`;
- all outputs and gradients finite;
- step 1: each semantic gamma received a nonzero gradient;
- step 2: deep projector and all three stage-gate weights received nonzero
  gradients;
- peak allocated CUDA memory: `1,733,776,384` bytes (about 1.61 GiB).

No dataset was read and no epoch training was started.

### 5.4 Pretrained load dry run

The existing ImageNet MXNet weight was converted and loaded into HST without
training:

- path:
  `/home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params`;
- size: `436,873,620` bytes;
- SHA256:
  `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`;
- converted keys: 191;
- unexpected keys: none;
- new HST/CAM-head keys: expected missing keys;
- backbone missing keys: only the already-audited public-baseline `bn45` and
  `bn52` affine/running-stat entries. HST introduces no additional missing
  backbone keys.

## 6. Parameters, FLOPs, memory, and runtime

Measured with `tools/profile_hst.py` on the 5090, batch 1, 224×224, 20 warmups
and 100 timed forwards:

| Metric | HFRM A0 | HST A1 | A1 vs A0 |
|---|---:|---:|---:|
| Total parameters | 112,709,714 | 107,537,234 | -4.589% |
| Rectifier parameters | 7,612,166 | 2,439,686 | -67.950% |
| Conv/Linear FLOPs per image | 200.488 GFLOPs | 200.478 GFLOPs | -0.005% |
| Median forward latency | 4.5783 ms | 4.5830 ms | +0.103% |
| Peak allocated CUDA memory | 601,748,992 B | 581,124,608 B | -3.427% |

FLOPs count only Conv2d and Linear operations and uses two FLOPs per
multiply-add. Pooling, normalization, sigmoid, and elementwise residual
operations are excluded consistently from both variants. The implementation is
comfortably within the specification's parameter-overhead limit; it actually
reduces total parameter count because the three large independent HFRM veto
MLPs are replaced by shared-width latent projections.

## 7. Experiment records and model selection

Each run writes:

- `experiment_config.json`: complete CLI arguments, model config, git commit,
  PyTorch/CUDA versions, parameter counts, and actual optimizer param groups;
- `eval_history.json`: seed, epoch, train loss/exact-match/accuracy, learning
  rate, validation/test mIoU and mDice, rectifier gammas, and best flag;
- `last.pth` (default checkpoint name);
- the final five epoch checkpoints;
- `best_val.pth`, selected **only by validation mIoU**.

Test evaluation occurs only after validation-based selection and is reporting
only. It cannot select a checkpoint or configuration.

## 8. A0/A1 commands and A2/A3 gates

### A0 frozen baseline

```bash
python train_sshr.py \
  --dataset bcss --rectifier hfrm --seed 42 \
  --trainroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/training/ \
  --valroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/val/ \
  --testroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/test/ \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --save_folder /home/duyanhong/mytestrepo-hst-a1/experiments/innovation1/A0_bcss_seed42
```

### A1 progressive-only

```bash
python train_sshr.py \
  --dataset bcss --rectifier hst --hst_variant a1 \
  --hst_latent_dim 256 --hst_context_kernel 15 --seed 42 \
  --trainroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/training/ \
  --valroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/val/ \
  --testroot /home/duyanhong/sshr-reproduction/SSHR/datasets/BCSS-WSSS/test/ \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --save_folder /home/duyanhong/mytestrepo-hst-a1/experiments/innovation1/A1_bcss_seed42
```

### A2 and A3

A2/A3 commands are not exposed yet. The parser intentionally rejects these
variants so an unreviewed architecture cannot be launched accidentally. After
A1 validation is reviewed, A2 will add the specified stage transition and
zero-initialized `rho`; only if A2 outperforms both A1 and A0 on validation will
A3 add an MLP hierarchy-token mixer.

## 9. Risks and next decision

- A1 is primarily a diagnostic. Since all stages decode the same correction
  state, it cannot test target-conditioned state evolution; that is the A2
  causal question.
- Zero residual scales intentionally delay inner semantic/context branch
  learning by one optimizer step. The two-step smoke confirms the paths open as
  expected.
- The public PolyOptimizer behavior, including the audited effective SGD
  momentum, remains unchanged to preserve the frozen baseline.
- The server runtime (PyTorch 2.11.0+cu128) differs from earlier reproduction
  environments, as explicitly permitted for the current 5090 workflow. A0 and
  A1 must therefore be compared on this same server/environment.
- Full A1 training should not begin until this milestone and its PR are
  reviewed. When approved, run A0 then A1 on BCSS seed 42 with all non-rectifier
  settings identical and select checkpoints by validation mIoU only.
