# Innovation 1 HST A3 Implementation Report

## 1. Scope and status

This milestone implements **A3 HLI + SCT**, the complete Innovation 1 code
path. It adds lightweight Hierarchical Latent Interaction (HLI) before the
already merged A2 Stage-Specific Correction-State Transitions (SCT).

Completed:

- a true cross-hierarchy four-token MLP mixer;
- residual HLI in the specified form `Z_hat = Z + Mixer(LN(Z))`;
- explicit A1/A2/A3 configuration defaults and an A3 identity control;
- raw/interacted token and HLI-residual diagnostics;
- A2 regression, cross-token dependency, gradient, finite-value, shape,
  optimizer-group, and full-forward tests;
- ten-step CUDA optimization-readiness smoke, efficiency profile, and
  pretrained-load audit on the RTX 5090 server.

Deliberately not performed:

- no A1, A2, or A3 full dataset training;
- no validation/test architecture selection;
- no attention or Mamba mixer;
- no change to A2 transitions or `rho=0.01` initialization;
- no loss, optimizer, CH, backbone, CAM-head, inference, or metric change.

A3 was implemented before full A1/A2 training at the user's explicit request.
This report therefore establishes implementation correctness and readiness,
not segmentation-quality improvement.

## 2. HLI architecture

The input order is fixed and retains hierarchy identity:

```text
Z = [z_deep, z_stage3, z_stage2, z_stage1]  # [B, 4, 256]
```

The A3 mixer is:

```text
normalized = LayerNorm(Z)
token_mixed = MLP_4x4(normalized across the four-token axis)
residual = Linear(256 -> 256) -> GELU -> Linear(256 -> 256)
           applied to token_mixed at each retained stage position
Z_hat = Z + residual
```

The explicit 4-token MLP is necessary: applying only a `256 -> 256` MLP to
each token independently would not let one hierarchy observe another. The
implementation is separable and lightweight: token mixing establishes
cross-level dependence, while the channel MLP performs the specified latent
semantic reorganization. No raw spatial feature is mixed.

The interacted descriptors then enter the unchanged A2 path:

\[
C_D=\phi_D(\hat z_D),
\]

\[
C_i=C_{i+1}+\rho_iT_i(C_{i+1},\hat z_i),
\qquad \rho_i=0.01\text{ initially}.
\]

Feature rectification remains:

\[
F_i^R=F_i+\gamma_i^{sem}S_i+\gamma_i^{ctx}CH_i(F_i).
\]

Because semantic `gamma` remains zero-initialized, the new HLI does not alter
the initial rectified features even though it reorganizes the latent tokens.

## 3. Variant isolation

| Variant | SCT | HLI default | Purpose |
|---|---:|---|---|
| A1 | disabled | identity | progressive-only diagnostic |
| A2 | enabled | identity | target-conditioned transition |
| A3 | enabled | MLP | complete HLI + SCT |
| A3 identity control | enabled | identity | exact A2 control through A3 interface |

The official `--hst_variant a3` command resolves and records `hli_mode=mlp`
and `transition_enabled=true` in `experiment_config.json`. A1/A2 reject MLP
interaction, and A3 rejects disabled transitions. The analysis-only A3 identity
control has the same parameter/state layout and exact tensor outputs as A2.

## 4. Files and diagnostics

| File | A3 responsibility |
|---|---|
| `network/hst/latent_interaction.py` | identity and separable cross-token MLP modes |
| `network/hst/hst_rectifier.py` | A3 defaults, HLI execution before SCT, diagnostics |
| `train_sshr.py` | A3 CLI and resolved experiment configuration |
| `tests/test_hst_a3.py` | A2 regression and A3 component/integration checks |
| `tools/smoke_hst_a3.py` | ten-step HLI/SCT optimization-readiness audit |
| `tools/profile_hst.py` | A3 parameters/FLOPs/runtime selection |
| `tools/check_pretrained_load.py` | A3 pretrained-load dry run |

`forward_with_diagnostics` now includes:

- `raw_latent_tokens` (`Z`);
- `latent_tokens` (`Z_hat`);
- `hli_residual` (`Z_hat - Z`);
- all previously available descriptors, correction states, transition deltas
  and scales, semantic gates, context/rectified features, and CAM logits.

## 5. Automated verification

Command:

```bash
python -m unittest discover -s tests -v
```

Result on Windows CPU and the RTX 5090 server: **30/30 passed**.

Key A3 checks:

- A1 identity HLI remains the same tensor with no parameters;
- A2 total parameter count remains exactly `109,505,621`;
- the A3 HLI has exactly `132,136` parameters;
- A3 identity-control outputs are bitwise equal to A2;
- an output hierarchy token has nonzero gradient with respect to other input
  tokens, proving real cross-token interaction;
- `Z_hat = Z + residual` holds exactly;
- HLI LayerNorm, token mixer, channel mixer, SCT, target projectors, and stage
  gates receive finite nonzero gradients when the semantic path is open;
- every trainable A3 parameter appears in exactly one optimizer group;
- full-network outputs and diagnostics are finite.

## 6. Ten-step CUDA optimization-readiness smoke

Environment:

- NVIDIA GeForce RTX 5090 D v2, 24 GB;
- Python 3.10.20;
- PyTorch 2.11.0+cu128;
- batch 2, 224x224 random input, ten optimizer steps;
- frozen SSHR classification-loss weights;
- no dataset access and no epoch training.

Result: **readiness passed at step 2**. Step 1 opens the zero-initialized
semantic gamma. At step 2, HLI, all transition MLPs, and all target hierarchy
projectors have finite nonzero gradients.

Triples below are ordered `stage3 / stage2 / stage1`. “Transition ratio” is
`||rho_i * delta_C_i|| / ||C_parent||`.

| Step | Loss | HLI grad | Token-mixer grad | Channel-mixer grad | HLI residual ratio | SCT MLP grad | Target-projector grad | Transition ratio | Finite |
|---:|---:|---:|---:|---:|---:|---|---|---|:---:|
| 1 | 0.6942 | 0 | 0 | 0 | 8.723% | 0 / 0 / 0 | 0 / 0 / 0 | 0.193% / 0.207% / 0.225% | Yes |
| 2 | 0.6918 | 1.12e-7 | 1.43e-8 | 1.11e-7 | 8.719% | 5.58e-9 / 5.63e-9 / 4.27e-9 | 3.12e-8 / 2.01e-8 / 5.22e-9 | 0.195% / 0.209% / 0.225% | Yes |
| 3 | 0.6902 | 2.16e-7 | 2.80e-8 | 2.14e-7 | 8.713% | 1.07e-8 / 1.08e-8 / 8.45e-9 | 5.98e-8 / 3.87e-8 / 1.00e-8 | 0.194% / 0.208% / 0.226% | Yes |
| 4 | 0.6895 | 3.10e-7 | 4.07e-8 | 3.07e-7 | 8.714% | 1.54e-8 / 1.56e-8 / 1.24e-8 | 8.59e-8 / 5.58e-8 / 1.45e-8 | 0.195% / 0.207% / 0.226% | Yes |
| 5 | 0.6886 | 3.94e-7 | 5.24e-8 | 3.91e-7 | 8.710% | 1.96e-8 / 1.99e-8 / 1.61e-8 | 1.09e-7 / 7.09e-8 / 1.85e-8 | 0.194% / 0.207% / 0.225% | Yes |
| 6 | 0.6879 | 4.70e-7 | 6.31e-8 | 4.65e-7 | 8.714% | 2.34e-8 / 2.37e-8 / 1.95e-8 | 1.30e-7 / 8.46e-8 / 2.20e-8 | 0.195% / 0.207% / 0.225% | Yes |
| 7 | 0.6865 | 5.34e-7 | 7.15e-8 | 5.29e-7 | 8.716% | 2.66e-8 / 2.70e-8 / 2.25e-8 | 1.48e-7 / 9.64e-8 / 2.50e-8 | 0.194% / 0.207% / 0.226% | Yes |
| 8 | 0.6865 | 5.87e-7 | 7.96e-8 | 5.81e-7 | 8.712% | 2.93e-8 / 2.97e-8 / 2.49e-8 | 1.62e-7 / 1.06e-7 / 2.76e-8 | 0.194% / 0.208% / 0.226% | Yes |
| 9 | 0.6856 | 6.29e-7 | 8.54e-8 | 6.23e-7 | 8.713% | 3.14e-8 / 3.19e-8 / 2.70e-8 | 1.74e-7 / 1.14e-7 / 2.97e-8 | 0.194% / 0.208% / 0.226% | Yes |
| 10 | 0.6850 | 6.58e-7 | 8.94e-8 | 6.52e-7 | 8.712% | 3.28e-8 / 3.34e-8 / 2.84e-8 | 1.82e-7 / 1.19e-7 / 3.11e-8 | 0.194% / 0.207% / 0.227% | Yes |

The initial per-token HLI residual ratios were 8.01% (deep), 8.17%
(stage3), 10.43% (stage2), and 7.97% (stage1). This is a moderate latent-space
residual, while SCT continues to modify each parent correction state by only
about 0.19% to 0.23%. Peak allocated CUDA memory was `1,853,116,928` bytes
(about 1.73 GiB).

Raw evidence:
`audit/results/hst_a3_cuda_smoke.json`, SHA256
`8c825afcae8a6e9cf4a9277c2a5286253f707d78314560a6d2484aa02572e6ff`.

## 7. Parameters, FLOPs, memory, and runtime

Measured on the RTX 5090 at batch 1, 224x224, with 20 warmups and 100 timed
forwards:

| Metric | HFRM A0 | HST A3 | A3 vs A0 |
|---|---:|---:|---:|
| Total parameters | 112,709,714 | 109,637,757 | -2.726% |
| Rectifier parameters | 7,612,166 | 4,540,209 | -40.356% |
| Conv/Linear FLOPs per image | 200.4880 GFLOPs | 200.4827 GFLOPs | -0.0027% |
| Median forward latency | 4.5794 ms | 4.6508 ms | +1.560% |
| Peak allocated CUDA memory | 601,748,992 B | 589,464,576 B | -2.041% |

A3 adds only `132,136` parameters over A2 (`+0.121%`) and approximately
`0.00053%` Conv/Linear FLOPs. It remains smaller than public HFRM and within the
Innovation 1 lightweight target.

## 8. Pretrained initialization audit

The same ImageNet MXNet weight loads into A3 without training:

- size: `436,873,620` bytes;
- SHA256:
  `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`;
- converted keys: 191;
- unexpected keys: none;
- expected missing keys: new HST/HLI/SCT and CAM-head parameters;
- missing backbone keys: only the previously audited public-code `bn45` and
  `bn52` affine/running-stat entries.

A3 introduces no additional missing backbone key.

## 9. Reproducible commands

Development verification:

```bash
python -m unittest discover -s tests -v
python tools/smoke_hst_a3.py --device cuda --batch_size 2 --image_size 224 \
  --steps 10 --output_json audit/results/hst_a3_cuda_smoke.json
python tools/profile_hst.py --device cuda --batch_size 1 --image_size 224 \
  --warmup 20 --iterations 100 --hst_variant a3
```

Future BCSS A3 run, documented but **not executed**:

```bash
python train_sshr.py \
  --dataset bcss --rectifier hst --hst_variant a3 \
  --hst_latent_dim 256 --hst_context_kernel 15 --seed 42 \
  --trainroot datasets/BCSS-WSSS/training/ \
  --valroot datasets/BCSS-WSSS/val/ \
  --testroot datasets/BCSS-WSSS/test/ \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --save_folder experiments/innovation1/A3_bcss_seed42
```

Checkpoint selection remains validation-mIoU-only. Test metrics are reporting
only and cannot select a variant, configuration, or checkpoint.

## 10. Decision boundary

A3 is implementation-complete, numerically ready, and lightweight. It has no
mIoU evidence because full training was intentionally not started. Therefore:

- do not claim A3 is better than A2;
- do not tune the HLI on validation or test before the frozen A0/A1/A2/A3
  comparison;
- use the A3 identity control if a future result must isolate HLI from SCT;
- retain the smallest empirically supported variant after validation.
