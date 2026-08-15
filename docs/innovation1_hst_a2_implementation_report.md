# Innovation 1 HST A2 Implementation Report

## 1. Scope and status

This milestone implements **A2 Stage-Specific Correction-State Transition**
on top of the merged A1 implementation. It preserves identity Hierarchical
Latent Interaction (HLI), so the only architectural difference between A1 and
A2 is the target-conditioned transition applied at stages 3, 2, and 1.

Completed:

- one independent transition module per target hierarchy;
- zero-initialized learnable residual scale `rho`;
- an explicit A1/A2 configuration and an A2 transition-disable control;
- A1-equivalence, formula, shape, gradient, parameter-group, finite-value, and
  full-forward tests;
- CUDA smoke testing, efficiency profiling, and pretrained-load auditing on the
  RTX 5090 server;
- reproducible A2 commands and experiment metadata support.

Deliberately not performed:

- no A1 or A2 full dataset training;
- no validation or test architecture selection;
- no A3 HLI mixer;
- no loss, inference, metric, backbone, CAM-head, CH, or optimizer change.

This follows the decision to implement A2 immediately while deferring the
costly A1 training run.

## 2. Formula-to-code mapping

For each target stage `stage3`, `stage2`, and `stage1`, A2 consumes the parent
correction state and the current hierarchy's latent descriptor:

\[
u_i=[C_{i+1}, z_i, C_{i+1}-z_i, C_{i+1}\odot z_i].
\]

The stage-specific MLP is:

```text
Linear(4d -> 2d) -> GELU -> Linear(2d -> d), d=256
```

The residual update is:

\[
C_i=C_{i+1}+\rho_iT_i(u_i), \qquad \rho_i=0\text{ at initialization}.
\]

This is implemented by `StageSemanticTransition` in
`network/hst/transition_block.py`. `HSTRectifier` applies the modules in the
fixed deep-to-shallow order:

```text
C_deep -> C_stage3 -> C_stage2 -> C_stage1
```

Only correction states are propagated. No raw backbone feature is passed from
one target stage to another. The stage gates, original SSHR Contextual
Homogenization branch, residual feature equation, and four CAM heads remain
unchanged from A1/A0.

HLI is explicitly `identity` in A2:

\[
\hat z_i=z_i.
\]

The parser rejects any non-identity HLI mode, preventing an accidental A3 run.

## 3. Configuration and controls

The public switches are:

| Variant | Transition | HLI | Meaning |
|---|---:|---|---|
| `hfrm` | n/a | n/a | unchanged public SSHR baseline |
| `hst/a1` | disabled | identity | progressive correction-state diagnostic |
| `hst/a2` | enabled | identity | target-conditioned stage transition |
| `hst/a2 --no-hst_transition_enabled` | disabled | identity | strict A1 control through the A2 interface |

`HSTConfig(variant="a1")` does not instantiate transition modules. This keeps
the merged A1 parameter count and state-dict structure unchanged. A2 adds three
independent transition blocks and exposes each `rho` through the existing
rectifier-scalar logging path.

## 4. Files changed

| File | A2 responsibility |
|---|---|
| `network/hst/transition_block.py` | exact target-conditioned residual transition |
| `network/hst/hst_rectifier.py` | A1/A2 configuration, top-down transition execution, diagnostics |
| `network/hst/__init__.py` | public transition export |
| `train_sshr.py` | A2 CLI/config plumbing and `rho` logging |
| `tests/test_hst_a2.py` | A2 component and integration regression suite |
| `tools/smoke_hst_a2.py` | dataset-free three-step training-path CUDA check |
| `tools/profile_hst.py` | selectable A1/A2 efficiency profile |
| `tools/check_pretrained_load.py` | selectable A1/A2 pretrained-load audit |

The diagnostics returned by `forward_with_diagnostics` now include raw and
interacted semantic descriptors, latent tokens, all four correction states,
three transition deltas, three `rho` values, gates, semantic/context features,
rectified features, and per-stage CAM logits.

## 5. Automated verification

Command:

```bash
python -m unittest discover -s tests -v
```

Result on Windows CPU and the 5090 server: **18/18 passed** (the original eight
A1 tests plus ten A2 tests).

The A2 suite verifies:

- `rho=0` returns the parent state with exact `torch.equal` equality;
- the implemented residual update exactly matches the A2 formula;
- zero-`rho` A2 produces the same correction-state progression as A1;
- disabling A2 transitions remains A1-equivalent even if `rho` is manually
  made nonzero;
- nonzero `rho` creates target-specific states using the correct stage target;
- all expected parameters receive finite gradients when residual paths are
  manually opened;
- every trainable A2 parameter belongs to exactly one optimizer group;
- full-network shapes, diagnostics, and outputs are finite;
- the A1 total parameter count remains exactly `107,537,234`.

## 6. CUDA smoke and zero-initialization behavior

Environment:

- NVIDIA GeForce RTX 5090 D v2, 24 GB;
- Python 3.10.20;
- PyTorch 2.11.0+cu128;
- batch 2, 224x224 random inputs, three optimizer steps;
- no dataset access and no epoch training.

Result:

- losses: `0.6959156`, `0.6937357`, `0.6931577`;
- all outputs and observed gradients finite;
- peak allocated CUDA memory: `1,760,631,296` bytes (about 1.64 GiB).

The observed activation sequence is important:

1. Step 1: zero-initialized semantic `gamma` receives gradient; gates,
   transition scales, target projectors, and transition MLPs receive zero
   effective gradient.
2. Step 2: after `gamma` opens, gates and `rho` receive nonzero gradients;
   transition MLPs and target projectors still receive zero gradient because
   `rho` was zero for this forward.
3. Step 3: after `rho` opens, transition MLPs and target projectors receive
   finite nonzero gradients.

The path is connected in FP32, but its initial inner-branch gradients are very
small. At step 3, target-projector norms were approximately
`4.95e-17` to `2.23e-16`, and transition-MLP norms were approximately
`1.56e-16` to `6.58e-16`. The learned `rho` values entering step 3 were only
about `1.53e-10` to `7.23e-10` in magnitude.

This is a genuine optimization risk caused by the specified nested zero gates
(`gamma=0` and `rho=0`), not a functional-disconnection bug. The implementation
does not alter either initialization because the A2 specification requires
both. A future training run must log `rho`, transition residual norms, and
correction-state similarity to determine whether the transitions learn at a
useful rate.

## 7. Parameters, FLOPs, memory, and runtime

Measured on the 5090 with batch 1, 224x224, 20 warmups, and 100 timed forwards:

| Metric | HFRM A0 | HST A2 | A2 vs A0 |
|---|---:|---:|---:|
| Total parameters | 112,709,714 | 109,505,621 | -2.843% |
| Rectifier parameters | 7,612,166 | 4,408,073 | -42.092% |
| Conv/Linear FLOPs per image | 200.4880 GFLOPs | 200.4816 GFLOPs | -0.0032% |
| Median forward latency | 4.5801 ms | 4.6206 ms | +0.8842% |
| Peak allocated CUDA memory | 601,748,992 B | 588,934,144 B | -2.129% |

A2 adds `1,968,387` parameters relative to A1, corresponding exactly to three
stage-specific transition blocks. It remains below both the original HFRM
parameter count and the specification's 5% overhead target. FLOPs count only
Conv2d and Linear operations, using two FLOPs per multiply-add.

## 8. Pretrained initialization audit

The existing ImageNet MXNet weight was converted and loaded into A2 without
training:

- size: `436,873,620` bytes;
- SHA256:
  `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`;
- converted keys: 191;
- unexpected keys: none;
- expected missing keys: new HST, transition, and CAM-head parameters;
- missing backbone keys: only the previously audited public-code `bn45` and
  `bn52` affine/running-stat entries.

A2 introduces no additional missing backbone key and does not change the
pretrained conversion logic.

## 9. Reproducible commands

A2 CUDA verification:

```bash
python -m unittest discover -s tests -v
python tools/smoke_hst_a2.py --device cuda --batch_size 2 --image_size 224
python tools/profile_hst.py --device cuda --batch_size 1 --image_size 224 \
  --warmup 20 --iterations 100 --hst_variant a2
```

Future controlled BCSS A2 run, documented but **not executed**:

```bash
python train_sshr.py \
  --dataset bcss --rectifier hst --hst_variant a2 \
  --hst_latent_dim 256 --hst_context_kernel 15 --seed 42 \
  --trainroot datasets/BCSS-WSSS/training/ \
  --valroot datasets/BCSS-WSSS/val/ \
  --testroot datasets/BCSS-WSSS/test/ \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --save_folder experiments/innovation1/A2_bcss_seed42
```

Checkpoint selection remains validation-mIoU-only. Test metrics are reporting
only and cannot select a configuration or checkpoint.

## 10. Decision boundary

A2 is implementation-complete and technically runnable, but there is no mIoU
evidence yet because full training was intentionally skipped. Therefore this
milestone proves correctness, compatibility, and cost; it does not prove the
scientific hypothesis.

The next scientifically valid comparison is A1 versus A2 versus the same A0
baseline under one frozen BCSS protocol. A3 must remain blocked until A2 has
been trained and reviewed. During A2 training, the nested-zero optimization
risk should be assessed before attributing a neutral result to the transition
hypothesis.
