# Innovation 1 HST A2 Implementation Report

## 1. Scope and status

This milestone implements **A2 Stage-Specific Correction-State Transition**
on top of the merged A1 implementation. It preserves identity Hierarchical
Latent Interaction (HLI), so the only architectural difference between A1 and
A2 is the target-conditioned transition applied at stages 3, 2, and 1.

Completed:

- one independent transition module per target hierarchy;
- fixed official A2 initialization `rho_3=rho_2=rho_1=0.01`, while `rho`
  remains learnable;
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
C_i=C_{i+1}+\rho_iT_i(u_i), \qquad \rho_i=0.01\text{ at initialization}.
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
| `hst/a2` | enabled, `rho=0.01` | identity | target-conditioned stage transition |
| `hst/a2 --no-hst_transition_enabled` | disabled | identity | strict A1 control through the A2 interface |

`HSTConfig(variant="a1")` does not instantiate transition modules. This keeps
the merged A1 parameter count and state-dict structure unchanged. A2 adds three
independent transition blocks and exposes each learnable `rho` through the
existing rectifier-scalar logging path. The official 0.01 initial value is a
code constant rather than a training CLI option, so this readiness fix does not
create a validation/test tuning dimension.

## 4. Files changed

| File | A2 responsibility |
|---|---|
| `network/hst/transition_block.py` | exact target-conditioned residual transition |
| `network/hst/hst_rectifier.py` | A1/A2 configuration, top-down transition execution, diagnostics |
| `network/hst/__init__.py` | public transition export |
| `train_sshr.py` | A2 CLI/config plumbing and `rho` logging |
| `tests/test_hst_a2.py` | A2 component and integration regression suite |
| `tools/smoke_hst_a2.py` | dataset-free ten-step optimization-readiness CUDA check |
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

Result on Windows CPU and the 5090 server: **20/20 passed** (the original eight
A1 tests plus twelve A2 tests).

The A2 suite verifies:

- `rho=0` returns the parent state with exact `torch.equal` equality;
- the implemented residual update exactly matches the A2 formula;
- the official A2 initialization is exactly 0.01 while every semantic `gamma`
  remains exactly zero;
- after manually setting all `rho` values to zero, A2 produces the same
  correction-state and rectified-feature progression as A1;
- disabling A2 transitions remains A1-equivalent even if `rho` is manually
  made nonzero;
- nonzero `rho` creates target-specific states using the correct stage target;
- all expected parameters receive finite gradients when residual paths are
  manually opened;
- every trainable A2 parameter belongs to exactly one optimizer group;
- full-network shapes, diagnostics, and outputs are finite;
- the A1 total parameter count remains exactly `107,537,234`.

## 6. CUDA optimization-readiness smoke

Environment:

- NVIDIA GeForce RTX 5090 D v2, 24 GB;
- Python 3.10.20;
- PyTorch 2.11.0+cu128;
- batch 2, 224x224 random inputs, ten optimizer steps;
- no dataset access and no epoch training.

The earlier three-step review with nested `gamma=0` and `rho=0` showed inner
transition/projector gradients around `1e-16`. The readiness patch keeps the
SSHR semantic `gamma` at zero but initializes the three learnable `rho` values
to 0.01. Transition MLP initialization is unchanged.

Result: **readiness passed**. All three transition MLPs and all three target
projectors had finite nonzero gradients at step 2. Their step-2 norms were
`6.79e-9` to `7.63e-9` and `2.44e-9` to `4.59e-9`, respectively: about seven
orders of magnitude above the original `1e-16` observation. All requested
finite checks passed at every step. Peak allocated CUDA memory was
`1,851,509,248` bytes (about 1.72 GiB).

Each triple below is ordered `stage3 / stage2 / stage1`. Values are sampled
before the optimizer update for that step. “Relative update” is exactly
`||rho_i * delta_C_i|| / ||C_parent||`.

| Step | gamma value | gamma grad | rho value | rho grad | MLP grad | target projector grad | relative update | Finite |
|---:|---|---|---|---|---|---|---|:---:|
| 1 | 0.00E+00 / 0.00E+00 / 0.00E+00 | 1.16E-03 / 8.56E-04 / 1.14E-03 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 0.00E+00 / 0.00E+00 / 0.00E+00 | 0.00E+00 / 0.00E+00 / 0.00E+00 | 0.00E+00 / 0.00E+00 / 0.00E+00 | 2.48E-03 / 2.42E-03 / 2.39E-03 | Yes |
| 2 | 1.16E-04 / -8.56E-05 / -1.14E-04 | 1.23E-03 / 7.76E-04 / 1.02E-03 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 9.09E-09 / 2.13E-09 / 1.04E-08 | 7.37E-09 / 7.63E-09 / 6.79E-09 | 4.59E-09 / 3.57E-09 / 2.44E-09 | 2.48E-03 / 2.43E-03 / 2.39E-03 | Yes |
| 3 | 2.28E-04 / -1.56E-04 / -2.07E-04 | 1.28E-03 / 7.03E-04 / 9.04E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 1.62E-08 / 4.23E-09 / 1.86E-08 | 1.35E-08 / 1.38E-08 / 1.23E-08 | 8.37E-09 / 6.46E-09 / 4.42E-09 | 2.48E-03 / 2.42E-03 / 2.39E-03 | Yes |
| 4 | 3.33E-04 / -2.14E-04 / -2.81E-04 | 1.33E-03 / 6.55E-04 / 8.08E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 2.11E-08 / 7.02E-09 / 2.63E-08 | 1.84E-08 / 1.87E-08 / 1.66E-08 | 1.15E-08 / 8.77E-09 / 5.97E-09 | 2.48E-03 / 2.43E-03 / 2.39E-03 | Yes |
| 5 | 4.29E-04 / -2.61E-04 / -3.40E-04 | 1.38E-03 / 5.93E-04 / 7.35E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 2.69E-08 / 7.16E-09 / 3.10E-08 | 2.24E-08 / 2.25E-08 / 2.00E-08 | 1.40E-08 / 1.06E-08 / 7.21E-09 | 2.48E-03 / 2.42E-03 / 2.39E-03 | Yes |
| 6 | 5.16E-04 / -2.99E-04 / -3.86E-04 | 1.41E-03 / 5.54E-04 / 6.69E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 2.94E-08 / 1.16E-08 / 3.54E-08 | 2.58E-08 / 2.56E-08 / 2.28E-08 | 1.61E-08 / 1.20E-08 / 8.21E-09 | 2.48E-03 / 2.42E-03 / 2.40E-03 | Yes |
| 7 | 5.92E-04 / -3.28E-04 / -4.22E-04 | 1.45E-03 / 5.12E-04 / 6.10E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 3.28E-08 / 9.89E-09 / 3.83E-08 | 2.83E-08 / 2.79E-08 / 2.48E-08 | 1.76E-08 / 1.30E-08 / 8.94E-09 | 2.48E-03 / 2.42E-03 / 2.39E-03 | Yes |
| 8 | 6.55E-04 / -3.51E-04 / -4.49E-04 | 1.47E-03 / 4.77E-04 / 5.65E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 3.27E-08 / 1.08E-08 / 4.10E-08 | 3.03E-08 / 2.96E-08 / 2.63E-08 | 1.88E-08 / 1.38E-08 / 9.50E-09 | 2.47E-03 / 2.43E-03 / 2.39E-03 | Yes |
| 9 | 7.05E-04 / -3.67E-04 / -4.68E-04 | 1.49E-03 / 4.43E-04 / 5.01E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 3.59E-08 / 1.27E-08 / 4.26E-08 | 3.18E-08 / 3.09E-08 / 2.74E-08 | 1.98E-08 / 1.45E-08 / 9.87E-09 | 2.47E-03 / 2.42E-03 / 2.39E-03 | Yes |
| 10 | 7.40E-04 / -3.77E-04 / -4.80E-04 | 1.51E-03 / 4.39E-04 / 4.91E-04 | 1.00E-02 / 1.00E-02 / 1.00E-02 | 3.79E-08 / 1.27E-08 / 4.50E-08 | 3.27E-08 / 3.16E-08 / 2.81E-08 | 2.04E-08 / 1.48E-08 / 1.01E-08 | 2.48E-03 / 2.42E-03 / 2.38E-03 | Yes |

The transition remains a small residual perturbation: the relative update is
stable around 0.238% to 0.248% over all stages and steps. `rho` remains close
to 0.01 through step 10 and is learnable. The raw per-step JSON, including
pre/post-update scalar values and every finite sub-check, is stored at
`audit/results/hst_a2_rho001_cuda_smoke.json` (SHA256
`d9608662dfadfb5841ebb376bf683f68e70b61f888c6ae2c811d1d9e483baf8b`).

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
python tools/smoke_hst_a2.py --device cuda --batch_size 2 --image_size 224 \
  --steps 10 --output_json audit/results/hst_a2_rho001_cuda_smoke.json
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

The A2 optimization-readiness check now passes: the transition path is active
by step 2 at a materially larger numerical scale while the actual correction
remains a small residual. The next scientifically valid comparison is A1 versus
A2 versus the same A0 baseline under one frozen BCSS protocol. A3 must remain
blocked until A2 has been trained and reviewed.
