# Method report — PRC Data Challenge 2026, team *optimistic-dragon*

Author: Daniel Hernández (MSc AI, University of Westminster). Repository: this folder (local git, GPL-3.0).
Final submission: `optimistic-dragon_final.parquet`, SHA-256 `f42b39e0…2216e2`, four months (Jan/Feb/Jun/Jul 2026).

## 1. Problem reading

`TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` for departures. Before modelling anything, the 2025 training data
were audited for how `BLOCK_TIME` is produced, because the ranking rows hide exactly that column and the target.
Three regimes were found (`src/v2_tail_analysis.py`, `outputs/experiments/v2/`):

| regime | share of 2025 DEP rows | mechanism | what the target equals |
|---|---:|---|---|
| 0 — physical taxi-out | 95.2 % | real off-block recorded | minutes; depends on airport, runway, congestion, NM plan |
| 1 — BLOCK = SCHED | 4.8 % (100 049 rows; 1 618 of the 1 676 rows > 1 h are LIRF) | the airport did not record the real off-block and the scheduled time was stored | exactly `MVT − SCHED`, tolerance ±10 s |
| 2 — one-day date shift | 13 rows (11 LIRF) | no NM off-block (`AOBT_3`) and departure on the calendar day after the schedule (local date) | ≈ 86 400 s + taxi |

The squared-error metric makes regimes 1 and 2 decisive: a single class-2 row mis-predicted costs
≈ 86 400² / 345 000 ≈ 21 600 MSE units, i.e. ≈ 35 s of RMSE on a 300 s baseline. The same artefact is visible on
the arrivals of the ranking files (where `BLOCK_TIME` is not hidden) with the same rates as in 2025 (LIRF Jan
8.7 % → 8.6 %, Jul 8.9 % → 10.6 %; other airports ± 0.3 pp), which is why modelling it was considered legitimate
and stable rather than a leak.

## 2. Model (V7)

### 2.1 Features
* V1 (`src/prepare_training.py`): calendar (hour, weekday, month, sin/cos), NM-plan deltas
  `MVT − {AOBT_3, EOBT_1, LOBT, IOBT, SCHED}`, `AOBT_3 − SCHED`, `has_aobt3`, congestion counts per airport-hour,
  airport-day, airport-runway-hour, arrivals per airport-hour; 10 categoricals (airport, runway, stand, aircraft
  type, market segment, flight type, wake category, operator, flight rule, destination).
* V2 (`src/v2_features.py`): windowed departure/arrival counts (±15/±30 min), runway spacing, time to next
  scheduled departure on the runway, relative congestion (fitted on the training block).
* V3 (`src/v3_protocol.py`): `days_after_sched = date(MVT) − date(SCHED)`.
* V4b (`src/v4b_features.py`): pairwise differences between NM timestamps, seconds-of-minute flags, previous
  arrival's taxi-in, delay and "copy" flag (arrival `BLOCK = SCHED`), copy rates of arrivals in ±60/±180 min,
  composite keys (airport×stand, airport×runway, airport×operator, route), flight-number frequency.
* Target encodings (`src/v5_lgb.py`): mean target per high-cardinality key computed **out of month** (each row's
  encoding excludes its own month; the prior excludes it too).

### 2.2 Components
* **Gate c1** — `v4b_protocol.GateClassifier`: CatBoost Logloss, 1 000 it., depth 6, negatives subsampled to 30 %
  with exact 1/frac weights (unbiased probabilities). Output `p1 = P(class 1 | x)`.
* **Cell classifier c0** — `v3_protocol.RegimeClassifier2`: CatBoost binary classifier for class 1 plus Bayesian
  cell counts for class 2 (pseudo-count α = 20) in the cell "date shifted ≥ 1 day and no AOBT_3", per airport, with
  a nested leave-one-out choice of the delay-band threshold inside the training block.
* **Normal branch b** — class-0 rows with 0 < y ≤ 3 h: CatBoost RMSE (depth 8, 3 000 it., lr 0.05;
  `v4b_protocol.NormalV4b`) and LightGBM (255 leaves, 2 000 it., lr 0.05, min 200 rows/leaf; `v5_lgb.LGBNormal`),
  averaged 50/50.
* **Combination** ("variant S", `v4b_final.combine`): outside the cell `pred = b + p1·(s − b)` with
  `s = MVT − SCHED`; inside the cell `pred = P0·b + P1·s + P2·(86 400 + b)`; clipped at 0. CatBoost components use
  seeds 42, 7 and 2025 and the three predictions are averaged.
* **LOBT-gap rule** (`v6_final.fit_lobt_rule / apply_lobt_rule`): if `MVT − LOBT > 7 200 s`,
  `pred = max(0, a + c·(MVT − LOBT))` with (a, c) = (−1 228.2, 1.0153) fitted by least squares on the 143 rows of
  2025 that satisfy the condition. Interpretation: once the aircraft is more than two hours past its last planned
  off-block the NM plan is no longer updated and the recorded off-block stays close to the plan, so the taxi time
  grows one-to-one with the gap. Tree models cannot extrapolate this.
* **V7c** (`v7c_noaobt.py`): `nb` = mean of `MVT − AOBT_3` (values in (0, 2 h)) over the other departures of the
  same airport within ±60 min; `nba = nb − median_2025(airport)`. For rows **without** their own AOBT_3, outside
  the cell and not covered by the LOBT rule, with `nba > 600 s`, the normal branch becomes `max(b, nb)`.
  Rationale: the model's main signal for those rows is missing and the trees fall back to the airport's typical
  value; during disruptions (snow at Schiphol, 3–9 Jan 2026) that is far too low.

### 2.3 What is deliberately not in the model
No external data (an OPDI/ADS-B probe was run and rejected: ground events cover ≈ 6 % of departures and predict
worse than the NM plan where present), no weather, no leaderboard feedback, no post-hoc per-row edits.

## 3. Validation protocol

* **Folds.** The ranking months are January and July, so the outer folds are *leave-January-out* (LOJO:
  train Mar–Dec 2025) and *leave-July-out* (LOJuO: train Jan, Mar–Jun, Sep–Dec). February and August are inner
  months: calibration, screening and any parameter choice happen there, never on the outer month.
  `RMSE_w = sqrt(0.4429·MSE_Jan + 0.5571·MSE_Jul)` mirrors the row shares of the ranking.
* **Uncertainty.** Paired block bootstrap by airport-day, stratified by month, 1 000 replicates; sensitivity to the
  three most influential rows is always reported, because the metric is dominated by a few rows.
* **Pre-registration.** Each step has a `protocol.md` with the acceptance rule written before the outer months
  were evaluated (e.g. V5: "ΔRMSE_w ≤ −1 s and improvement in both months"; V7: "improvement in total and in the
  affected rows in every month with ≥ 30 affected rows, and bootstrap CI upper bound < 0"). Rejected variants stay in
  the repository with their numbers.
* **Leaderboard.** Scores were recorded once in `submissions/upload_manifest.json` and never used to select or
  tune. Three uploads were made in the leaderboard phase and one in the final phase.

### 3.1 Results of the steps (Jan/Jul 2025 unless stated)

| step | ΔRMSE_w | 95 % CI | notes |
|---|---:|---|---|
| V3 → V4b (new normal branch + gate) | −14.6 s | [−25.7, −0.8] | gate −10.0, branch −4.6; 10/10 airports improve; the pre-set −15 s threshold was *not* met, V4b was uploaded as a second submission anyway (documented) |
| V4b → V5a (LightGBM blend) | −1.4 s | [−2.0, −0.9] | both months improve; robust to single rows |
| V5a → V6 (LOBT rule) | −11.5 s | [−34.9, −0.35] | Jan −0.7, Jul −21.3; inner months −0.4; almost all from a few rows; threshold 1.5 h/3 h give −10.2/−10.8 |
| V6 → V7 (V7c) | −0.35 s | [−0.86, −0.01] | Jan −0.74 (112 rows), Feb −8.84 (173 rows); Jul/Aug have < 30 affected rows |

Leaderboard (Jan + Jul 2026): V3 317.99, V4b 321.80, V6 286.58. The expected-RMSE model built from the gate's own
probabilities predicted 294 for V4b with a ±1σ luck band of [265, 319]; the realised 321.8 and the fact that V4b
lost to V3 despite the validated −14.6 s illustrate how much the ranking score depends on a few rows.

### 3.2 Error budget of the final model on the 2025 validation months (out-of-fold)

| rows | share of rows | share of squared error | RMSE within group |
|---|---:|---:|---:|
| class 0, normal (y ≤ 1 h, error ≤ 25 min) | 95.0 % | 44 % | 200 s |
| class 0 with y > 1 h | 0.2 % | 32 % | 3 636 s |
| classes 1 and 2 (artefacts) | 4.7 % | 16 % | 530 s |
| class 0, y ≤ 1 h, error > 25 min | 0.1 % | 8 % | 2 583 s |

Where the remaining gap to the top of the leaderboard lies: the LIRF artefacts (the gate reaches AUC 0.93 on rows
without an NM plan; what is left is near-irreducible from the available columns) and the long physical taxi-outs
under disruption, for which 2025 offers few examples.

## 4. Final phase (four months)

The organisers published `final_ranking.parquet` (1 341 197 rows; 670 790 departures across Jan/Feb/Jun/Jul 2026)
and `final_submitting.parquet` on 14 October, 1 h 20 min before the deadline. Procedure (`src/final_build.py`):

1. January/July rows: identical to `candidate_v7.parquet` (the MVT_IDs are the same as in the leaderboard file).
2. February/June rows: the V7 design re-trained on the 12 months of 2025 with seed 42 — LightGBM normal branch,
   LightGBM gate (`v5_gate.LGBGate`, validated as equivalent to the CatBoost gate in the V5 screening),
   `RegimeClassifier2`, LOBT rule, V7c — predicting the new months' features (built from the new file restricted to
   those two months, so congestion/arrival context comes from the correct days).
3. Assembly against the template (row order, dtypes, 670 790 rows), SHA-256 recorded, upload with remote hash
   verification. The evaluator accepted the file (`Succeeded`, 670 790 pairs).

Deviation from the leaderboard-phase model, stated plainly: for February/June the normal branch is LightGBM only
(the 3-seed CatBoost branch takes ≈ 2 h and could not be retrained in time). The validated value of the 50/50
blend is 1.4 s of RMSE. Everything else is the same code path.

## 5. Reproducibility and integrity

* Deterministic seeds (42; 42/7/2025 for the CatBoost ensemble). `candidate_v6` and `candidate_v7` were rebuilt on
  a second machine with byte-identical SHA-256; feature tables were checked column-by-column by hash between
  machines.
* The original Parquet files are never written; nothing derived from the data, no model and no credential is
  tracked by git. Uploads require an interactive credential prompt and verify the remote hash.
* Tests (`pytest tests`, 60 tests): out-of-month target encoding has no leak, regime labels are never computed
  for ranking rows, the LOBT rule touches only rows above the threshold, V7c never lowers a prediction, the
  neighbour statistic excludes the row itself and other airports, scripts contain no upload commands.

## 6. Limitations

* The approach depends on the `BLOCK = SCHED` and date-shift artefacts persisting in 2026; the arrivals of the
  ranking files indicate they do, but the final-phase months were not inspected for this before submission.
* Gains of the last steps (V6, V7) are concentrated in few rows and were partly discovered by inspecting all of
  2025 before pre-registering their coefficients; the report says so where it applies.
* The final models are not persisted as files; predictions are. A production version would save the bundles
  (the V1 code already does this: `models/final/<model>/model.cbm + manifest.json`).
