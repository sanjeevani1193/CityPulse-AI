# ExperimentLab: synthetic station redistribution experiment

ExperimentLab is a reproducible statistical demonstration, **not an actual Citi
Bike intervention**. It does not read real rides, station100 panels or test
forecasts. All `SIM0000`–`SIM0099` station IDs, baseline counts and post outcomes
are generated locally. The existing experiment package previously contained only
MLflow import utilities; those remain unchanged. NumPy and SciPy already existed
in the main environment; no local dependencies were installed.

## Design fixed before analyzing outcomes

| Item | Specification |
| --- | --- |
| Question | Would a hypothetical redistribution policy increase accepted recorded station starts? |
| Unit / assignment | 100 simulated stations, complete balanced randomization: 50 treatment, 50 control; assignment seed 42 |
| Treatment | Hypothetical station redistribution; control is unchanged service |
| Primary metric | Station post-period accepted starts divided by 28 local-calendar days |
| Estimand | Station-average intention-to-treat effect in recorded starts/station/day |
| Hypothesis | Two-sided H₀: mean effect = 0, α=0.05 |
| Primary estimator | Pre-calibrated CUPED difference in station means, Welch t interval/test |
| Sensitivity estimator | Unadjusted difference; not a second opportunity to declare a win |
| Practical threshold | +2 starts/station/day; practical benefit requires primary lower 95% bound >2 |
| Injected effect | Exactly +3 starts/station/day (+84 period starts) for treated stations |
| Stopping | Fixed 100 stations and 28-day post period; no peeking, adaptive stopping or subgroup search |

Simulation seed 2026 generates stationary lognormal baseline daily rates
(log-SD 0.35, population mean 80). Four nonoverlapping periods have Poisson totals
conditional on each station's rate: days [0,28), [28,56), [56,84), [84,112).
The first two are calibration covariate/outcome; the third is the experiment's
baseline covariate; the fourth is the untreated potential post outcome. The
constant injected count increment defines the treated potential outcome. These
are abstract local-calendar-day periods, not timestamped historical rides or
24-elapsed-hour forecast buckets. No DST/hourly construction is implied.

Randomization is independent of potential outcomes, with seeds recorded. Each
station contributes **one** analysis row. The simulation never treats repeated
station-hours or individual rides as randomized units. All units are observed;
no missing values are imputed as zero. Invalid/missing inference inputs fail.

## Methods and causal assumptions

Welch's two-sided t-test and Satterthwaite degrees of freedom operate across
stations, allowing unequal arm variances. SciPy supplies p-values and confidence
intervals; tests also verify these against independent manual equations. These
are approximate large-sample/normal-theory intervals, not exact randomization
intervals. Balance tables include pre-period arm means, SDs, a standardized
mean difference and a diagnostic p-value; a balance p-value is not a randomization
guarantee and never changes inclusion or estimator choice.

CUPED uses `Y_adjusted = Y − theta * (X − mean(X))`. Theta is covariance(calibration
X, calibration Y) / variance(calibration X), estimated **only from two earlier
pre-treatment periods** on all stations, then frozen before assignment/outcomes.
This differs from the common implementation that fits a pooled coefficient on
experimental post outcomes; the pre-only approach makes the leakage boundary
particularly explicit. Conditional on observed pre-treatment data, this is a
fixed adjustment independent of assignment. Group means on the adjusted scale
are not newly observed ride counts. Tests change the treatment effect and
assignment while requiring identical theta/preprocessing.

Variance reduction is `1 − SE_adjusted² / SE_raw²` for the **effect estimator**,
not a promise about every future dataset. It is allowed to be negative. The
simulation's extremely stable station rates make pre-history highly predictive;
real time trends and spillovers can reduce or eliminate the benefit.

Power uses pre-treatment calibration SDs, a two-sided noncentral-t calculation
with equal arm sizes/variances, and an 80% target. Raw planning SD is calibration
outcome SD. Adjusted planning SD is calibration residual SD (n−2 degrees of
freedom). This is a plug-in planning approximation; it ignores uncertainty in
estimated SD/theta and does not substitute for network/cluster power analysis.
No observed-post-effect “observed power” is reported. Symmetric survival tails
avoid a measured SciPy tiny-tail `NaN` issue without replacing the distribution
with a normal approximation.

Sources: [SciPy Welch test and confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html),
[SciPy noncentral t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html),
[CUPED adjustment and fixed-coefficient rationale](https://docs.datadoghq.com/experiments/statistics/cuped/).

## Actual measured example

Open the self-contained [HTML report with interval and station-scatter visualizations](experimentlab_report.html)
in a browser (download/open locally when viewing on GitHub).

| Estimator | Control mean | Treatment mean | Effect (starts/station/day) | 95% CI | Two-sided p |
| --- | ---: | ---: | ---: | --- | ---: |
| Unadjusted | 81.8264 | 84.6350 | +2.8086 | [−10.0947, 15.7118] | 0.66666 |
| CUPED primary (adjusted means) | 81.8542 | 84.6072 | +2.7531 | [1.7190, 3.7871] | 8.03e−7 |

Measured effect-estimator variance reduction: **99.358%**, not a real operational
result. The primary estimate supports a positive simulated effect but its lower
bound is below +2: **practical significance remains uncertain**. The unadjusted
result is inconclusive. This does not establish physical demand, recovered lost
rentals, profitability or real redistribution effectiveness.

| Pre-treatment planning | SD | Power for +2 at 50/arm | 80%-power MDE | Required stations/arm for +2 |
| --- | ---: | ---: | ---: | ---: |
| Raw | 33.0798 | 6.03% | 18.7193 | 4,296 |
| CUPED | 2.3884 | 98.56% | 1.3516 | 24 |

Two sets of 200 seeded Monte Carlo replicates (null and +3 effects) gave:

| True effect | Raw rejection | CUPED rejection | CUPED CI coverage | Mean CUPED estimate |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.0% | 5.5% | 94.5% | 0.0359 |
| +3 | 8.5% | 100% | 94.5% | 3.0359 |

Replicates are independent within each scenario; null/known-effect scenarios
reuse seeds for paired comparisons. Exact binomial Monte Carlo intervals are
saved: null rejection 2.78%–9.63%; known-effect rejection 98.17%–100%.
Thus 200/200 detection is not proof of perfect power. CI coverage interval is
90.37%–97.22% in both scenarios. Do not extrapolate these checks beyond this DGP.

## Reproduce and recover

Application: `src/citypulse_ai/experiments/redistribution.py`; standalone HTML
renderer: `experiments/report.py`; tests: `tests/test_experimentlab.py`.
Existing Python 3.14 runtime: NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1.
For a fresh environment, `ci/experimentlab-requirements.txt` pins the tested
statistics-only dependencies (SciPy 1.18.1 requires Python >=3.12). The optional
`experimentlab` project extra also supports compatible older SciPy on Python 3.11;
that alternative dependency resolution is not the release's tested runtime.

```sh
# Existing environment; no dependency installation necessary on this Mac.
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_experimentlab.py -q
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m citypulse_ai.experiments.redistribution \
  --output outputs/experimentlab/my-new-run
# Open outputs/experimentlab/my-new-run/report.html in a browser.
```

The output directory must not exist. It contains `experiment.json` (design,
station records, estimates, planning, simulation checks, package versions and
implementation source hashes), `report.html` and a SHA-256 checksum manifest.
No earlier report is overwritten. Seeds/config and code are sufficient to
reconstruct the entirely synthetic inputs; outputs stay ignored. The reviewed
18 kB HTML example is intentionally committed as a small synthetic report,
without real station records, credentials or machine-specific paths.

An exact reproduction of the final analysis, Monte Carlo intervals and HTML
measured 0.56 seconds and 98,926,592-byte process max RSS (~94 MiB). All three
ignored output files together occupy 42,186 bytes. The regenerated results and
HTML matched saved artifacts exactly, and both checksums verified. No heavy ML
runtime or Docker process is imported by ExperimentLab.

## Release verification and limitations

Nineteen focused tests cover seeded assignment, identity integrity, exact injected
effect, station-level inference equations, pre-only calibration, power/MDE,
invalid/missing data, degenerate variance, null/known-effect simulations,
create-only outputs/checksums and standalone report generation. The dedicated
[ExperimentLab CI](https://github.com/sanjeevani1193/CityPulse-AI/actions/workflows/experimentlab-ci.yml)
runs on standard Ubuntu with Python 3.14 and uploads only a synthetic report.
Existing inference CI verifies artifact-free gateway/worker contracts; SQLPilot
CI verifies offline tests and its real container tool/UI path without Groq.
Those jobs do not retrain models or rerun held-out/live benchmarks.

The coherent demonstration is complete once these release jobs pass. Actual
operational adoption still requires a preregistered trial, ethically/practically
viable assignment, interference-aware units, cost/service/safety guardrails,
station-status/completeness evidence, and a powered study. Authentication/public
production hosting, weather integration, expanded SQLPilot evaluation, optional
QLoRA/vLLM and future infrastructure remain outside this three-component release.
