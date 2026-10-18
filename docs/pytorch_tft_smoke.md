# Native PyTorch TFT development smoke test

This implements the architecture in the [station TFT design](tft_station_forecasting_design.md)
using the existing [station windows](station_dataset_prototype.md) and
[positive-label policy](station_training_labels.md). It is an architecture and
training-plumbing milestone, not final training, a benchmark or evidence of
predictive accuracy. No new dependencies or model-serving changes are needed.

## Architecture

`forecasting/tft.py` contains the following components:

| Component | Purpose |
| --- | --- |
| Station embedding and static variable selection | Encode frozen station codes; code 0 is reserved, selected codes are 1–100 |
| Four static-context GRNs | Condition temporal variable selection, static enrichment, and initial LSTM hidden/cell states |
| Scalar embeddings and separate historical/future variable selection | Transform each feature separately and weight it with contextual softmax selection |
| Gated residual networks (GRNs) | ELU transformations, optional static context, GLU gates, residual projections and layer normalization |
| LSTM encoder and decoder | Encode 336 elapsed historical hours; initialize the known-future decoder from encoder states |
| Gated local skip and static enrichment | Preserve selected local inputs while enriching recurrent features with station context |
| Interpretable temporal attention | Separate query/key projections per head, one shared value projection, and averaged heads |
| Positionwise GRN and gated residual output | Combine attention with the local recurrent representation |
| Ordered quantile head | Predict q10/q50/q90 jointly for all 24 elapsed target hours |

The model receives **no future demand, target arrays or target-quality masks**.
Encoder counts and operating-status values with false input masks are replaced
by finite placeholders internally; masks and age remain distinct input features.
Missing counts do not remove elapsed-hour positions, since their calendar is
still known. Attention can use those calendar/missingness representations; it is
not claimed that every attended hour has a demand observation.

Attention queries cover decoder positions only. Each query can attend to all
encoder positions and decoder positions through its own step, never later
decoder representations. The decoder LSTM is also unidirectional. Tests perturb
later future-known covariates and show earlier outputs are unchanged. All 24
calendar vectors are known at issuance; the causal restriction is conservative.

Simplifications are explicit: one static categorical variable (so its static
selection weight is necessarily 1), continuous scalar temporal embeddings,
one LSTM layer per encoder/decoder, no static geography/weather embeddings,
and decoder-only attention queries instead of computing unused historical
query outputs. Historical variable selection includes the 11 known features
and 5 observed/mask features; decoder selection includes only the 11 known
features. Dropout is 0.1 for the real smoke test and disabled by `eval()`.
Selection/attention weights are diagnostics, not causal explanations.

For raw head outputs `(a, m, b)`, normalized quantiles are
`(m − softplus(a), m, m + softplus(b))`. This monotone parameterization prevents
crossings by construction. It does not establish calibration or interval
coverage. Inverse affine scaling preserves order. Forecasts are not clipped;
negative lower quantiles remain possible and require an explicitly chosen
count-support treatment before production or benchmarking.

## Tensor contract and preprocessing

| Tensor | Shape |
| --- | --- |
| Static station code | `(B, 1)` integer |
| Encoder known calendar/position | `(B, 336, 11)` |
| Encoder observed count/mask/age/status/status mask | `(B, 336, 5)` |
| Decoder known calendar/position | `(B, 24, 11)` |
| Separate normalized target and boolean mask | Each `(B, 24, 1)` |
| Normalized or inverse-scaled quantile forecasts | `(B, 24, 3)` |

`forecasting/tft_preparation.py` reuses per-station scalers fitted on **unique,
trusted training station-hours**, not repeated windows or masked placeholders.
It rejects corrupt trusted training counts. The existing window builder checks
scaler training bounds against the declared split. Only count values are
scaled, not masks or calendar features. Targets remain in original units in the
dataset contract and are transformed only at trusted positions in the batch
adapter. `inverse_quantiles` returns original ride-count units.

Preprocessing is saved separately as a small JSON file with exact station-ID
mapping, feature order, means, positive scales, sample counts and fitting bounds.
Loading validates the feature schema, station mapping and scaler parameters.
The model file contains configuration and state tensors, loaded with CPU mapping
and `weights_only=True`; the runner refuses to overwrite existing diagnostics.

## Masked loss

For each trusted target, compute the three pinball terms
`max(q * (y − prediction), (q − 1) * (y − prediction))`, average across quantiles,
sum across trusted station-hours and divide by the number of trusted targets.
Thus masks cannot dilute loss through a larger batch/grid denominator.

Masked targets are indexed out **before loss arithmetic**, including masked
NaNs. No missing observation becomes a zero-demand label. An all-masked batch
returns `None`, and `training_step` skips backward and the optimizer step,
including weight decay. Finite loss/gradients are checked, gradient norm is
clipped at 1, and parameters are checked after updates.

## Reproduce the bounded real smoke test

From the repository root, use the existing environment and a new output folder:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/smoke_tft.py \
  --label-review outputs/station-label-review/dev-20261015-verified/report.json \
  --cohort-report outputs/station-dataset-prototype/dev-20261015-verified/report.json \
  --output-dir outputs/tft-smoke/new-diagnostic-run
```

The runner rechecks local historical hashes/checkpoints and re-adjudicates only
the existing June 1 and July 1 two-station slices. Evidence identities must match
the prior review. This reads validated records and existing hourly checkpoints;
it does not rerun historical reconciliation, source ingestion or cloud work.
Each two-station slice has only 720 grid rows. The station IDs remain `6140.05`
and `5905.14`; no new cohort selection occurs.

The fixed development configuration is hidden size 16, two attention heads,
62,511 parameters, seed 42, CPU with one PyTorch thread, Adam at 0.001 and
**exactly three optimizer steps**. June 1's two windows provide **11 trusted
target hours out of 48** (5 and 6 per station) and 66 usable encoder hours.
Scalers use **77 unique trusted hours**, 32 for `6140.05` and 45 for `5905.14`,
from May 18 04:00 UTC through June 2 04:00 UTC exclusive. The two target windows
belong to the permitted January–June development training interval; fitting
scalers on these training targets is permitted. These sparse, fixed-slice
scalers are diagnostic, not final full-training-period preprocessing.

July 1 is loaded **after all optimizer steps**, only for inference. It never
chooses preprocessing, stopping, architecture or hyperparameters. Its two
windows contain 11 trusted targets, but no validation metric is computed or
used. The frozen Jan–Jun development cohort is known at June end, so training
on earlier dates is retrospective architecture development, not a claim that
this cohort was available for a historical June forecast.

Measured report: `outputs/tft-smoke/dev-20261015/report.json`.
The three normalized masked losses were **0.326886, 0.351429, 0.347587**;
pre-clipping gradient norms were **1.153358, 1.434616, 1.177763**. All were
finite, and parameters changed. These noisy three-step plumbing checks do not
show convergence or forecast skill. July output shape was `(2, 24, 3)` in original
ride units, with finite values and no quantile crossings. Repeated inference
and saved-state reload matched exactly.

The measured run used Python 3.14.7, PyTorch 2.14.1 and pandas 3.0.6. Three
training steps took **0.286 seconds**; the entire run before reporting took
**17.594 seconds**, including offline provenance rechecks. Peak process RSS was
**369,229,824 bytes (~352 MiB)**. The diagnostic state is 383,911 bytes and
preprocessing JSON 922 bytes; the report is also small. These measurements
describe this batch/configuration only, not final 100-station training resources.
DuckDB review queries retain their 512 MB ceiling with spill disabled.

## Tests, label coverage and next steps

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_tft.py tests/test_station_dataset.py \
  tests/test_station_history.py tests/test_station_labels.py
```

The focused new tests exercise 100-code vocabulary support, forward/backward
passes, static contexts, selection weights, shared-value causal attention,
masked input invariance, loss values/normalization, all-missing optimizer skip,
finite gradients, inference determinism, checkpoint/preprocessing reload,
original-unit inversion, train-only scaler invariance and spring/fall DST
windows with exactly 24 elapsed targets. The broader relevant suite also covers
the existing station preparation and synthetic historical regressions.
Final verification passed **132 tests in 19.26 seconds**, including 18 new TFT
tests. Syntax and whitespace checks passed; the full ML suite was not run.
A separate fresh Python process restored the saved model and preprocessing and
reproduced all 144 real-data quantile outputs exactly, without additional training.
The three diagnostic files total 394,072 bytes (392 KiB allocated locally).
Host free space was 26,115,596,288 bytes before and 26,076,672,000 after the smoke
run; the host-wide change is not attributed entirely to these small artifacts.

The label policy is **unchanged and conservatively global across selected
stations for each rejected-start UTC hour**. Even a rejection apparently tied to
another station blocks all selected stations in that hour. An unlocalizable
rejected start masks the entire requested slice. This may discard valid counts;
it avoids guessed attribution. The June/July slices each retain only a fraction
of their labels, zero demand is still uncertified, and none of these windows is
eligible for complete-24-target primary evaluation. Positive-only training is
biased and must not be presented as all-hour demand accuracy.

Before final training: complete the source history and boundary review, approve
count coverage/zero/rejection policies, freeze the final 100-station training-only
cohort, fit full permitted training scalers, implement chronological training and
validation with checkpoint selection, test calibrated count-supported quantiles,
and define fair common evaluation keys against existing models. No final TFT
training, model serving integration or benchmark is implemented in this phase.
