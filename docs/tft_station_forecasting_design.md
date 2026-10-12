# TFT and station-level forecasting design

Design reviewed against the repository and local metadata on 2026-10-12. This document proposes a future implementation; no dataset was built, model trained, dependency installed, or cloud resource accessed for this work.

## Forecasting contract

Forecast **recorded, validated ride starts**, in rides per station per elapsed hour, for a fixed cohort of 100 NYC stations. This is realized usage, not unconstrained demand: empty docks, closures, and bike shortages can suppress starts.

Issue one forecast per station at each **America/New_York local midnight**. Convert the issuance instant `o` to UTC, then predict buckets `[o+h hours, o+(h+1) hours)` for `h=0,…,23`. Store lead as `horizon_step=1,…,24`; step 1 is the bucket starting at issuance. All 24 predictions use the same information cutoff.

| Split | NYC-local interval, end exclusive | UTC interval, end exclusive |
| --- | --- | --- |
| Train | 2024-01-01 to 2025-07-01 | 2024-01-01 05:00Z to 2025-07-01 04:00Z |
| Validation | 2025-07-01 to 2025-10-01 | 2025-07-01 04:00Z to 2025-10-01 04:00Z |
| Test | 2025-10-01 to 2026-01-01 | 2025-10-01 04:00Z to 2026-01-01 05:00Z |

A sample belongs to a split only when its entire 24-hour target interval lies within that split. Drop a crossing origin with a recorded reason; do not truncate its horizon or move targets between splits. Encoder history can precede a split boundary. Completed validation observations can inform later validation/test forecasts; completed test observations can inform later test forecasts. They cannot update fitted weights, station selection, or preprocessing statistics.

## Existing code: reusable parts and limits

| Component | Reuse | Required extension |
| --- | --- | --- |
| `src/citypulse_ai/etl/trips.py` | String station IDs, UTC timestamps, audited rejections, sparse station-hour counts | Quality/coverage assessment before treating accepted counts as complete labels; DST ambiguity remains unresolved |
| `src/citypulse_ai/etl/historical.py`, `history_catalog.py` | Source hashes, row reconciliation, boundary provenance, cross-archive duplicate audit, start-month canonicalization | Full source coverage and a verified final canonical snapshot; distinguish retrospective labels from information available at issuance |
| `src/citypulse_ai/features/preparation.py` | Dense UTC grid, local calendar, nullable labels, coverage flags, causal optional zero heuristic | Configurable final dates, fixed cohort, per-hour source quality, availability timestamps, window builder; current constants and SQL table are February 2025 only |
| `forecasting/baselines.py` | Explicit timezone boundaries, elapsed lag-24/168, source-before-origin checks, metrics | Station grouping, availability checks, masks, DST-aware evaluation manifest |
| `forecasting/experiment.py` | Immutable read-only loading, hashes, duplicate rejection and boundary reconciliation | Current loader is January–February 2024 citywide, using January–March archives; retain station dimension and parameterize the final contract in new adapters |
| `forecasting/xgboost_model.py`, `mlp.py` | Known target calendar, safe historical lags, training-only scaling/selection patterns | Current nine-feature citywide adapters and short split constants cannot directly consume station panels or masks |
| `forecasting/lstm.py` | Midnight sequence origins, UTC contexts, 24 outputs, train-only scaler, split containment | Current input is 48 univariate hours and rejects every missing context/target; future adapter needs longer masked station sequences and known future covariates |
| Existing preparation/model tests | Fixtures for missingness, future invariance, lag provenance, boundaries, scaling | Full-period DST, cohort freezing, issuance latency, masked losses and model-independent scoring |

The preparation code's `coverage_intervals` and `activity_periods` are retrospective audits. Full-window first/last observations, longest gaps, and episode endpoints must not become forecast inputs or determine causal zero fills. The existing `training_station` flag means “seen in training,” not “selected stable station.” Its complete-grid station universe also includes later stations; the final modeling grid must use only the frozen cohort.

## Current data readiness

Read-only inspection of `outputs/historical/manifest.json` found processed source archives **2024-01, 2024-02, 2024-03, 2025-02, and 2025-03**. The manifest has no registered `canonical` entry, and no canonical report was found under `outputs/historical/canonical/`. This does not establish the availability of other external copies. It does establish that the inspected local manifest is insufficient for the final 24-month dataset and 18-month cohort selection.

The existing `outputs/phase21-forecasting/report.json` records February 2025 preparation: 2,237 stations, 1,503,264 station-hour rows, 598,630 observed station-hours, and **904,634 unknown station-hours**, with zero station-hours filled. These are existing report values, not new measurements or proof that stations were closed. Its source-completeness confirmation is for that preparation run, not the final history.

Existing January 2026 boundary-probe metadata contains December 2025 starts in sampled CSV prefixes. It is evidence that archive month differs from start month; it is not full January 2026 ingestion or complete December coverage. No verified operating-status history, issuance-time trip feed, historical weather forecasts, station metadata snapshot, or TFT dataset is established by the inspected preparation code/metadata. Readiness is **partial exploratory data and reusable contracts, not a training-ready final panel**.

## Training-only station selection

Make cohort selection a separate reproducible artifact, computed only from canonical starts in the training interval. Do not use validation/test counts, future station survival, or full-window coverage reports.

Proposed fixed rule, to approve before inspecting final validation/test demand:

1. Preserve IDs as exact strings, including leading zeros and reviewed suffixes. Do not merge IDs by station name or coordinates without an independently justified, versioned identity map available by training end. Use NYC archive scope; never add Jersey City `JC-` archives. Any finer geographic filter requires a documented metadata rule.
2. Require training-source quality sufficient to evaluate all 18 local months. Missing source months block selection; they are not station inactivity.
3. Define an active local date as at least one accepted start. Require at least **15 active dates in each of the 18 training months** and no run exceeding **14 consecutive local dates without recorded starts**, including leading/trailing training gaps. These are proposed activity-stability proxies, not operating-status evidence.
4. Rank qualifying stations by total accepted training starts, descending; break ties by exact station ID in ascending lexical order. Select the first 100.
5. Save all candidates, training-only monthly/date statistics, exclusion reasons, ranks, selected IDs, thresholds, source hashes, and training cutoff. Encode selected IDs deterministically as 1–100; reserve 0 for unknown IDs outside this primary task.

These thresholds are the proposed acceptance specification, not measured eligibility results. Selection is accepted only when all 18 training months pass source-quality review and at least 100 stations qualify. If fewer than 100 qualify, stop cohort finalization, report the eligible count and exclusion reasons, and obtain agreement on any revised training-only thresholds before rerunning selection. Do not silently relax thresholds or substitute future-active stations. Never fill the cohort with stations chosen for strong future demand or replace stations that decline during evaluation. Retain all selected stations and report their scored coverage.

This cohort is retrospective within training: its stability is known at training end, not January 2024. That is valid for learning a fixed cohort deployed after training. It cannot support a claim that January 2024 forecasts used a cohort selected contemporaneously in January.

## Observation and availability semantics

Keep three separate concepts: **archive/source coverage**, **valid count observation**, and **station operating availability**. The existence of one ride proves activity at that instant; it does not prove a whole hour was observed completely or that the station stayed operational.

| Situation | Count and label policy | Availability policy |
| --- | --- | --- |
| Positive accepted count, complete trusted coverage | Count is a usable label | Unknown unless independent status evidence covers the interval |
| Sparse positive count, incomplete source or relevant rejection uncertainty | Preserve accepted count for audit; complete target is unknown | Unknown |
| No row, complete trustworthy station-hour count enumeration | A zero can represent **recorded starts**, even if operating status is unknown; document the enumeration guarantee | Keep operating status unknown |
| No row, archive merely marked processed, station-hour coverage unproven | Unknown count, not zero | Unknown |
| Independently verified operating station and complete count source, no starts | Valid zero label | Known operating |
| Known closure but incomplete trip source | Do not manufacture a count label from status alone | Known unavailable |

A complete enumeration guarantee must establish source membership, late/boundary contributions, duplicates, and relevant rejection quality; a globally reconciled accepted-row total alone does not prove complete real ride-start counts. Define whether labels mean all physical starts or starts accepted under the frozen validation policy, and document exclusions. The proposed benchmark uses the latter, with uncertain hours masked.

**Proposed trusted-zero acceptance policy:** a missing station-hour row becomes `ride_count=0`, `label_observed_mask=1`, with reason `trusted_zero_complete_enumeration` only when a versioned coverage decision certifies that hour for the frozen station ID. The decision must verify all required source members and their recorded checksums, input = accepted + rejected reconciliation, accepted-count conservation, global duplicate handling, NYC-local start-month boundary contributions (including relevant later archives), and the absence of unresolved rejection/timestamp uncertainty affecting that hour. Save the source set, audit references, target validation policy, review cutoff, and certified UTC intervals. Reconciliation alone or `status=processed` is insufficient. If any prerequisite is incomplete or uncertain, keep the missing count null and its label mask 0; do not treat a missing archive as a zero-start interval.

For a certified complete enumeration, zero means no starts accepted under the declared target policy. It does **not** mean the station was open. Unknown operating status stays null with `availability_known_mask=0` and does not veto an otherwise trusted realized-start zero. Conversely, known closure does not certify count-source coverage. Coverage evidence finalized retrospectively may adjudicate scoring labels, but becomes a historical input only under the explicitly declared retrospective replay assumption below.

Conservative first version retains unknown counts unless the coverage review justifies zeros. A positive-only labeled panel would omit quiet hours and bias training/scoring toward busy periods. It is useful for readiness diagnostics, but **cannot establish an all-hour station-demand benchmark**. Resolving trustworthy zero labels is a prerequisite for that claim.

The existing `station_zero_horizon` fills zeros after recent activity, causally and with an assumption flag. Leave it disabled for the primary dataset. A future sensitivity dataset could retain these as `assumed_zero_recent_activity`; do not call them observations or mix their scores with the primary policy. Never fill before the first observation, interpolate from future starts, or infer a closure from silence.

## Proposed feature schema

Use a long station-hour panel plus a sample manifest and immutable metadata. Separate scoring labels from forecast inputs so unavailable decoder demand cannot enter feature construction.

| Field | Type / grain | Role and rule |
| --- | --- | --- |
| `station_id`, `station_code` | String and integer, static | Group key and learned categorical embedding, frozen from training |
| Optional coordinates / borough | Static snapshot | Add only with documented identity and as-of provenance; absent from the minimal design |
| `hour_start_utc`, `time_idx` | UTC timestamp and int64, station-hour | Unique `(station_id, hour_start_utc)`; shared elapsed index from dataset start; sorting/alignment, not an arbitrary scaled date feature |
| `hour_start_local`, `utc_offset_minutes`, `dst_fold` | Derived station-hour | Local audit timestamp; offset and fold distinguish repeated clock hours |
| `ride_count_accepted` | Nullable integer, station-hour | Sparse audit count before coverage adjudication |
| `ride_count`, `label_observed_mask` | Nullable nonnegative integer and bool | Trusted observed target, including justified zero; missing stays null in storage |
| `source_coverage_status`, `label_quality_reason` | Enum/string, station-hour | Complete, partial, ambiguous, or unknown with evidence; audit metadata |
| `station_available`, `availability_known_mask` | Nullable bool and bool, station-hour | Independent status; unknown is not unavailable; past inputs only when known at origin |
| `observation_available_at_utc`, `status_available_at_utc` | Nullable UTC timestamps | When each input became accessible, distinct from event time; unknown publication time must not be invented |
| `coverage_status_available_at_utc` | Nullable UTC timestamp | When source-quality evidence was known; a retrospective complete-archive flag is not automatically a historical input |
| `hour`, `weekday`, `is_weekend`, `month` | Local derived values | Future-known calendar; weekday Monday=0, month 1–12; encode hour/weekday/month cyclically |
| `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos` | Float, station-hour | Calendar encodings with periods 24, 7, and 12; month angle uses month minus 1 |
| `is_dst`, `utc_offset_hours`, `dst_fold` | Numeric/bool, station-hour | Future-known timezone features, derived from UTC and pinned timezone rules |
| Optional `is_holiday` | Bool, station-hour | Only with a versioned calendar known at issuance; exclude from minimal version |
| `origin_utc`, `origin_local_date`, `horizon_step` | Sample / target-step | Forecast key and lead; split is audit/selection metadata, not a predictive feature |
| `input_count_mask`, `input_age_hours` | Origin-dependent encoder step | Count accessible at issuance; elapsed time since most recent accessible trusted count, capped at 336, with 336 used when none exists |
| `encoder_length`, `decoder_length`, exclusion reason | Sample manifest | Fixed 336/24, eligibility and reproducibility |

`input_count_mask` can differ from `label_observed_mask`: an eventual trustworthy label can still be unavailable at midnight. Future label masks, realized status, audit completeness, rejection counts, and actual demand are reserved for scoring/loss, never decoder inputs.

Minimal calendar vector has **11** numeric channels: six cyclic values, weekend, `is_dst`, UTC offset in hours, `dst_fold`, and relative position in the 360-step encoder/decoder block. Relative position is deterministic and avoids fitting a scaler to future dates. Raw calendar values remain in storage for audit and tabular adapters.

## Encoder lookback and tensor examples

Recommend **336 elapsed hours (14 × 24)**. Two weeks contain two weekly cycles and all target-aligned lag-168 source hours, with more context for short missing runs than a single week. This does not guarantee two identical local weeks around DST. Local calendar features explain clock shifts. A later validation-only comparison may consider 168 and 336; longer history increases memory and warm-up exclusions and is not automatically better.

For issuance `o`, the encoder is `[o−336h, o)` and the decoder is `[o, o+24h)`. With no approved pre-2024 history, skip initial training origins lacking the full encoder span; do not download December 2023 or fabricate padding history. Internal missing hours stay on the elapsed grid with masks. A 336-hour window can contain missing observations without changing sequence length.

Example for batch size `B=64`, one station-origin pair per sample:

| Tensor | Shape | Contents |
| --- | --- | --- |
| `static_station_code` | `(64, 1)` | One selected categorical ID |
| `encoder_known` | `(64, 336, 11)` | Calendar and relative-position vector |
| `encoder_observed` | `(64, 336, 5)` | Scaled count, input count mask, capped age / 336, status value, status-known mask |
| `decoder_known` | `(64, 24, 11)` | Known calendar/position only |
| `target`, `target_mask` | Each `(64, 24, 1)` | Original count labels / trusted-label indicator, outside feature inputs |
| Quantile predictions | `(64, 24, 3)` | Proposed q10, q50, q90 |
| Point predictions | `(64, 24)` | q50 after inverse transform and declared postprocessing |

Unknown numeric encoder counts use a finite placeholder of 0 **after scaling**, accompanied by mask 0; a true zero has mask 1 and its actual transformed value. Unknown status uses value 0 and known-mask 0. These placeholder values are tensor mechanics, not labels. Keep missing targets null on disk and safely replace them only inside a masked loss calculation; multiplying a NaN loss by zero is insufficient.

Fit one count scaler per selected station from unique trusted training station-hours only, excluding unknown/assumed counts and avoiding repeated-window weighting. Store mean, positive scale, sample count, and bounds; use scale 1 for constant series. Reuse these frozen statistics for encoder/target transforms and inverse predictions. Do not use encoder- or decoder-window target normalization that learns from future values. Masks/bools remain unscaled; calendar cycles are already bounded.

Proposed sample eligibility: full time span in the approved dataset, at least one accessible encoder count, and all 24 trusted labels for the primary comparable evaluation subset. For future TFT training, partial labels can contribute via masked loss if at least one target is valid; normalize by the valid-target count. Produce predictions for every structurally eligible station-origin independently of target validity. Freeze an input-only coverage rule using training before evaluation; separately report origins with no usable history. Do not choose coverage thresholds for favorable test scores.

## Timezone, DST, and archive boundaries

Generate local dates/midnights using timezone-aware calendar-day operations, then convert each origin to UTC. Generate encoder and target hours using UTC and elapsed `Timedelta(hours=…)`. Never construct successive origins by adding 24 elapsed hours, strip timezones, or collapse the two fall-back 01:00 buckets.

For the planned years, spring transitions occur on March 10, 2024 and March 9, 2025; fall transitions occur on November 3, 2024 and November 2, 2025. These dates follow the America/New_York timezone rules and should be asserted in future fixtures. For example, a 2025-03-09 midnight forecast spans through the bucket beginning **2025-03-10 00:00 EDT**. That hour is also the first target of the next midnight forecast: one-hour overlap. A 2025-11-02 midnight forecast ends at **2025-11-02 23:00 EST**, leaving the 23:00–00:00 bucket outside that day's forecast; the next forecast starts at November 3 midnight. This is the intentional consequence of 24 elapsed hours issued on 23/25-hour local days.

Keep forecast records keyed by **station, origin UTC, target UTC, lead**. Score issued forecasts; the overlapping hour is two different forecast decisions, not a duplicate to silently remove. Report overlap/gap counts and optionally a separately defined unique-target diagnostic. Do not present summed 24-hour forecasts as local calendar-day totals on transition days.

Current ETL localizes naive NYC times with ambiguous/nonexistent values rejected. Explicit offsets can identify repeated instants; naive fall-back 01:xx cannot. Rejected starts, or end-time ambiguity causing otherwise identifiable starts to be rejected, can undercount accepted labels. Review these audits and mark affected hours uncertain under a documented policy; when an ambiguous start could belong to either fold, both candidate buckets need review. If rejection scope cannot be localized, widen the quality exclusion. Never infer folds from unsorted trips or assume rejection-free complete coverage. Preserve the rejection policy in the target definition.

Archive filenames describe publication/source membership, not necessarily ride-start month. Reuse canonical trip provenance and global duplicate audits before aggregating start-month station counts. Do not simply concatenate and sum source-hour files when ride identities overlap. Include all approved sources contributing in-window starts, including reviewed adjacent archives. January 2026 is required boundary input for December 2025; one adjacent month does not prove that no later archive contains long-trip starts. Review longer spillover and record any coverage limitation. January 2024 boundary review and each intervening month require the same discipline. Pre-window starts and 2026 starts stay outside modeling labels; source hashes and boundary audits remain in provenance.

## Forecast issuance and weather leakage

An event timestamp before midnight is necessary but insufficient for input eligibility. Require its hourly bucket to have ended by `o` and its observation version to have been accessible by `o`. In practice, an hour ending at midnight may not yet be finalized. Use the latest permissible as-of version or mask it; do not read a revised eventual count into that forecast. Do not select starts using a trip end, rejection decision, or later duplicate discovered only after issuance as though that information were already available.

Monthly historical trip archives cannot reproduce a live midnight feed. **The initial final benchmark is designated a RETROSPECTIVE forecasting experiment.** Its declared zero-latency assumption is that the eventually finalized, quality-approved count for `[t,t+1h)` is accessible at `t+1h`, including the hour ending exactly at midnight. This is an assumption about replay inputs, not a measured publication timestamp; preserve actual publication timestamps as unknown when unavailable. Coverage adjudication follows the same idealized assumption, while future target counts and masks remain excluded from inputs. Therefore distinguish two future datasets:

- **Retrospective exploratory replay:** explicitly assumes finalized counts are available at the end of each hour. This reuses current experiments' idealized assumption, and can compare models under that identical assumption. Record it prominently; do not call it an operationally verified backtest.
- **Operational replay:** requires actual observation versions/publication timestamps, status snapshots, and latency. Builds the encoder as of each origin, including delayed recent hours as missing. A fixed delay can be a declared sensitivity assumption, but is not measured feed availability.

Optional observation-latency sensitivity: repeat the retrospective replay with fixed delays `L=1` and `L=3` elapsed hours, allowing a historical bucket only if `t+1h+L <= origin`. Mask delayed encoder values and recompute lag availability and history summaries; never move the origin or target horizon. Freeze cohort, labels, splits, and feature definitions; fit new model adapters on training inputs subjected to the same delay and use validation only for selection. Report each delay's input exclusions and paired scores, plus a common-key intersection across delays for direct comparison. These are assumption-based sensitivity runs, not operational backtests.

Build a separate input view for every origin. Later revisions, target-hour actuals, future status/quality masks, and backward filling must not change that origin's inputs. Retrospective labels may be finalized later for scoring; keep this separation explicit.

Exclude weather from the initial shared feature set. Historical realized temperature/rain can be encoder observations only if accessible by issuance. Future weather requires a forecast archive keyed by issue time, valid time, location, and revision; choose only forecasts issued **and accessible** by midnight. Tomorrow's observed weather, reanalysis, or a later forecast revision is not future-known. Without suitable forecast vintages, omit decoder weather. Any weather-enhanced experiment must give comparable models the same permitted vintage and handle missing forecasts with origin-known masks.

## TFT architecture

TFT combines several components rather than replacing recurrence with attention:

1. **Static encoders** turn station embeddings into contexts for selection, enrichment, and recurrent state initialization.
2. **Variable selection networks** separately weight transformed static, historical, and future-known variables. Their weights can vary by station and time.
3. **Gated residual networks** apply nonlinear transformations, optional context, gates, skip connections, and normalization. Gates regulate the contribution of a processing block.
4. **LSTM encoder/decoder** processes history and then known future covariates; the decoder starts from encoder states. Decoder demand actuals are excluded.
5. **Static enrichment and attention** combine station context with temporal representations. Interpretable multi-head attention links useful time positions; causal masking prevents access to later temporal representations. TFT shares value projections across attention heads.
6. **Multi-horizon head** maps each decoder representation to quantile forecasts. Masked pinball loss supports missing labels; all 24 outputs belong to one issuance decision.

These are the components described in [Lim et al., sections 4–5](https://arxiv.org/html/1912.09363v3). Attention and selection weights offer diagnostics, not causal explanations or a guarantee of weekly-pattern learning. The proposed CityPulse outputs are q10/q50/q90; validate coverage, crossing, and calibration separately. No advantage over the existing models is assumed.

## One dataset for TFT and existing model families

Keep storage independent of a TFT library. Build the station-hour panel once, then create model-specific views from the same origin manifest and as-of input rules:

| Family | Adapter view |
| --- | --- |
| Seasonal naive | Station series with source hours `target−24h` and `target−168h`; require both time and availability checks |
| XGBoost / MLP | One station-origin-target row; station encoding, future calendar, allowed lag values/masks, and encoder-derived summaries; all demand sources must precede origin and be available then |
| LSTM | Masked station encoder, station embedding and optional future calendar branch, with 24 direct outputs; a univariate version remains a separately labeled reduced-input baseline |
| TFT | Static station codes, masked historical values, future-known calendar, 24 masked labels |

Lag-24/168 means elapsed-hour lags and changes local-clock alignment across DST. A same-local-hour previous-day/week baseline is a distinct optional design with an explicit ambiguous-hour mapping, not a silent change to current baselines. With delayed observations, lag-24 sources for late leads can be unavailable even though they precede issuance; the common eligibility manifest must account for this.

A model with masks may use inputs that the current strict missing-data adapters reject. Produce an input-eligible intersection for the primary paired comparison, independent of model errors, and report how much data it excludes. Also report TFT's wider masked coverage separately. Do not silently drop poor predictions, omit declining cohort stations, or claim identical training examples when architectures have different usable data. A controlled comparison can train all families on the same eligible origin blocks; a wider-data comparison must label that difference.

Citywide views should aggregate the **full eligible station universe**, under the same source-quality policy, not just the selected 100. A sum over selected stations is a cohort aggregate and must be named accordingly. TFT marginal quantiles cannot be summed to obtain calibrated aggregate quantiles without a dependence model. Existing citywide checkpoints/results are exploratory artifacts at another grain and split; future station adapters require new fits in a later authorized phase. Preserve prior models/results.

## Dataset-building strategy and leakage controls

The following is a future build plan, not actions executed for this design:

1. Verify approved immutable local inputs and manifest hashes. Review coverage, boundary starts, rejection/DST quality, and global duplicates. Block final preparation on unresolved required sources; never zero-fill a missing archive.
2. Produce or consume an approved canonical start-month snapshot with unique station-hour keys. Separate source/label quality adjudication from input availability. Record target definition and assumptions.
3. Select and save the training-only cohort. Fit categorical vocabularies and count statistics only on training; freeze these before validation. Save an ID/source/contract hash.
4. Construct a dense hourly UTC grid for the cohort over the final interval. Join sparse counts and independently justified zeros/status; retain missing values, masks, and provenance. Derive local calendar from UTC.
5. Generate all local-midnight origins and 336/24 elapsed windows. Assign splits by target containment, record warm-up/cross-boundary exclusions, and create origin-specific input views using the availability policy.
6. Persist a sample manifest, separate labels, ordered feature specification, transformations, and quality/eligibility reports. Suggested future artifacts: `station_hourly.parquet`, `station_cohort.parquet`, `forecast_origins.parquet`, `forecast_targets.parquet`, `preprocessing.json`, and `dataset_report.json`, under a new gitignored run directory.
7. Materialize windows lazily from the panel instead of writing repeated 336-hour arrays. One global model shares parameters across stations; batches mix training station-origin samples after chronological membership is fixed. Correlated windows are not independent evidence.

No full-period aggregates, validation/test demand statistics, future availability, or target-derived normalization enter cohort choice or features. Random row splits are prohibited. Centered rolling summaries and backward interpolation are prohibited. Shifted summaries are computed only from accessible pre-origin counts with an explicit valid-count denominator; missing values never become implicit zeros.

Use validation for limited model/configuration selection and early stopping; test has no role in fitting or selection. Freeze the training-only model for the declared final comparison. Any later train-plus-validation refit needs a separately defined protocol and cannot revise the cohort with validation demand. Keep exploratory January–February scores separate from final station-level results.

## Fair evaluation

Every comparator uses the same cohort, split, midnight origins, target buckets, latency assumption, zero policy, masks, and original count units. Save predictions by station/origin/target/lead and join labels afterward. A future primary report should include:

**Evaluation acceptance rules:** freeze the primary paired manifest before inspecting model errors. Its unique key is `(station_id, origin_utc, target_hour_start_utc, horizon_step)`; accept a station-origin block only when all 24 target labels are trusted and every comparator passes its predeclared input eligibility rule. Each model must provide exactly one finite prediction for every frozen key, with no missing or extra keys; an inference failure fails the paired run rather than shrinking the manifest. Validate equal key sets and joined actuals before calculating metrics. The cohort artifact must contain exactly 100 unique IDs in deterministic rank order and its training-only provenance must pass review; do not finalize it in this ingestion phase.

Report trusted-label numerators and denominators over the entire 100-station elapsed-hour grid, by station and split, separately from scored forecast-record coverage (whose denominator includes DST overlap). Separate positive labels, trusted zeros, unknown labels, and assumed zeros; assumed zeros are excluded from the primary labels. Report candidate origin blocks, warm-up/split-boundary exclusions, input-ineligible blocks, incomplete-label blocks, and accepted paired blocks, with overlapping exclusion reasons explicitly identified. Require at least one accepted paired block for every cohort station in each validation/test split to claim a complete 100-station paired benchmark. If this fails, retain the cohort, name the unscored stations, and report a partial-coverage diagnostic instead. No minimum percentage is silently imposed or tuned using evaluation demand; all coverage fractions accompany the results.

- Pooled MAE, RMSE, and WAPE over identical trusted forecast records; WAPE is undefined if summed actual demand is zero.
- Per-station MAE and its unweighted mean, to expose whether busy stations dominate pooled scores; per-station WAPE only where its denominator is positive.
- Errors by horizon, weekday, and DST period; scored station/origin/target counts, unknown-label fractions, input exclusions, and overlap/gap counts.
- q10/q50/q90 pinball losses and q10–q90 coverage/width for TFT. Use q50 as its declared point forecast; MAE and RMSE need not favor the same point estimator.

Apply consistent declared nonnegativity clipping after inverse transformation for point forecasts; record quantile crossing handling and do not tune it on test. Quantile training versus MSE is another comparison difference. Distinguish a shared-information architecture comparison from the existing reduced-input LSTM baseline. Any uncertainty estimate should respect day/week dependence and simultaneous station errors, rather than treating hourly records as independent samples. Do not claim improvement or significance without measured final evaluations.

## Concrete implementation steps and acceptance checks

Only a later authorized phase should implement these changes:

1. Add configurable contract/schema and cohort helpers under `src/citypulse_ai/features/`, leaving educational preparation defaults intact. Test exact ID preservation, deterministic ranking, insufficient-cohort failure, and invariance to arbitrary validation/test demand changes.
2. Add station-panel/quality preparation and a forecast-window builder in that package. Test unknown versus zero, partial positive counts, leading/trailing gaps, unavailable revisions, masked numeric placeholders, count conservation, and source-boundary duplicates with small synthetic fixtures.
3. Add final task adapters under `src/citypulse_ai/forecasting/`. Test source-before-origin and source-available-by-origin for every lag/summary, full target split containment, training-only scaler statistics, and future-target/status perturbation invariance.
4. Test spring/fall UTC windows, repeated local hours, missing local hours, 24 outputs, daily-origin overlap/gaps, archive boundaries, warm-up exclusions, and prior-split encoder use. Include a transition-adjacent split fixture even though the fixed July/October boundaries are not transition dates.
5. Add shared evaluation joins and masked losses. Test missing labels cannot enter inputs, all-missing losses are skipped safely, partial-label denominators are correct, zero denominators are reported, and all paired models score the exact same keys.
6. After dataset acceptance, choose and separately authorize a TFT implementation/dependency. Verify its normalization, automatic missing-timestep handling, decoder target access, attention masks, and custom target-mask loss against this contract. Do not assume a library's defaults enforce these rules.

This design was verified by reading source code, existing tests, local JSON reports, and the primary TFT architecture paper. No forecasting tests or benchmarks were run: the sole change is this document. Final dataset/model validation remains future work, blocked chiefly by incomplete registered history, trustworthy zero-label coverage, and the absence of operational as-of observation evidence.
