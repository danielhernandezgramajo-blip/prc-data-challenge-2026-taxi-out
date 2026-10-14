# PRC Data Challenge 2026 — taxi-out time prediction (team *optimistic-dragon*)

Reproducible pipeline that predicts `TAXITIME_SEC_mvt` (seconds from off-block to take-off) for departure
movements (`PHASE_mvt == "DEP"`) at the 10 airports of the challenge, trained only on the 2025 training files.
Official metric: RMSE. Licence: GPL-3.0 (`LICENSE`). Language: code comments and experiment protocols are in
Spanish; this README and `docs/METHOD.md` are in English for the review.

## Final submission (phase 2, four months)

| | |
|---|---|
| File uploaded | `optimistic-dragon_final.parquet` → bucket `prc-2026-optimistic-dragon`, 2026-10-14 09:18:55 UTC |
| Evaluator | `Succeeded`, 670 790 pairs against `prc-2026-datasets/final_submitting.parquet` |
| SHA-256 | `f42b39e02eb7fd09912ed701e2f03f71d632d9b2f389e208eb41dc46aa2216e2` (= `submissions/candidate_final.parquet`) |
| Model | **V7** (see below). January/July rows are identical to `candidate_v7.parquet`; February/June rows were predicted by the same V7 design re-trained on the 12 months of 2025 (`src/final_build.py`). No 2026 data enters any fit. |

Leaderboard phase (January + July 2026, 344 841 rows), recorded once and never used to tune anything:

| upload | model | RMSE (s) |
|---|---|---:|
| `optimistic-dragon_v1` (2026-10-01) | V3 (regime mixture with date-shift cells) | 317.99 |
| `optimistic-dragon_v2` (2026-10-02) | V4b (new normal branch + multivariate gate) | 321.80 |
| `optimistic-dragon_v3` (2026-10-06) | **V6** (V5a + LOBT-gap rule) | **286.58** |

## What the model does (V7)

The target is not a single physical quantity. In the 2025 data three regimes coexist, all of them functions of
columns that are also present in the ranking files:

* **Class 0 — physical taxi-out** (≈ 95 % of rows): off-block time is real; taxi-out is minutes.
* **Class 1 — `BLOCK_TIME = SCHED_TIME`** (≈ 4.8 %, mostly LIRF): the airport did not record the real off-block
  and the field was filled with the scheduled time, so the target equals exactly `MVT_TIME − SCHED_TIME`
  (often hours).
* **Class 2 — one-day date shift** (≈ 10 rows/year, LIRF): no NM off-block and a flight that departs on the
  calendar day after its schedule; the target is ≈ 86 400 s + taxi.

V7 predicts the conditional expectation over the three regimes and then applies two physically motivated
overrides:

```
b   = 0.5 · CatBoost_normal + 0.5 · LightGBM_normal            # class-0 branch, trained on 0 < y ≤ 3 h
p1  = P(class 1 | x)                                            # CatBoost gate (V4b), 3 seeds averaged
cell = date(MVT) > date(SCHED) and no AOBT_3                    # LIRF-dominated cell
pred = cell ? P0·b + P1·(MVT − SCHED) + P2·(86 400 + b)          # 3-class mixture (RegimeClassifier2, nested cell counts)
            : b + p1 · ((MVT − SCHED) − b)
pred = max(pred, 0)
V7c : rows with no AOBT_3 and neighbour congestion > 600 s  →  b := max(b, mean taxi proxy of neighbours)
LOBT: rows with MVT − LOBT > 2 h                            →  pred := max(0, −1228.2 + 1.0153 · (MVT − LOBT))
```

* **Features** (`src/prepare_training.py`, `v2_features.py`, `v3_protocol.py`, `v4b_features.py`): calendar,
  NM-plan deltas (`MVT − AOBT_3/EOBT_1/LOBT/IOBT/SCHED` and their pairwise differences), congestion counts per
  airport-hour/day/runway, relative congestion, arrival context (previous arrival's taxi-in and delay, copy rate of
  arrivals), date shift, stand/runway/operator/route keys. Target encodings are computed out-of-month inside the
  training block only.
* **Gate** (`v4b_protocol.GateClassifier`): CatBoost binary classifier with 30 % negative subsampling and
  re-weighting; calibrated p1 feeds the soft mixture ("variant S").
* **Normal branch**: CatBoost (depth 8, 3 000 it.) and LightGBM (255 leaves, 2 000 it., out-of-month target
  encoding) trained on class-0 rows ≤ 3 h; averaged 50/50 (`src/v5a_final.py`).
* **LOBT-gap rule** (`src/v6_final.py`): when the aircraft departs more than 2 h after its last planned off-block
  the NM plan is stale; in 2025, 143/143 such rows had taxi-out > 1 h and follow `y ≈ MVT − LOBT − 1 000 s`.
  Trees cannot extrapolate this slope-1 relation; a straight line fitted on those 143 rows can.
* **V7c** (`src/v7c_noaobt.py`): rows without their own AOBT_3 during extreme congestion (neighbour taxi proxy
  > 10 min above the airport median) are systematically under-predicted; their normal branch is raised to the
  neighbours' mean. No fitted parameter.

## Validation protocol (how every decision was made)

* Two outer folds that mimic the ranking months: **LOJO** (train Mar–Dec, evaluate January 2025) and **LOJuO**
  (train Jan + Mar–Jun + Sep–Dec, evaluate July 2025); February/August serve as inner months for calibration and
  screening. Metric: `RMSE_w = sqrt(0.4429·MSE_Jan + 0.5571·MSE_Jul)` (weights = share of rows in the ranking).
* Uncertainty: paired block bootstrap by airport-day (1 000 replicates); single-row influence checks.
* Every candidate had an **acceptance rule written down before the outer months were looked at**
  (`outputs/experiments/v*/protocol.md`). Rejected candidates are kept in the repository.
* Leaderboard scores were recorded once in `submissions/upload_manifest.json` and never used to choose or tune.
  The organisers' notice about exploiting the ranking process was followed literally: 3 uploads in the
  leaderboard phase, 1 in the final phase.

| step | change | validated gain (RMSE_w, Jan/Jul 2025) | 95 % CI |
|---|---|---:|---|
| V2 → V3 | regime mixture + date-shift cells | pooled dev −23 s, December −7 s | P(improve) 1.00 |
| V3 → V4b | new normal branch + multivariate gate | −14.6 s | [−25.7, −0.8] |
| V4b → V5a | 50/50 CatBoost + LightGBM normal branch | −1.4 s | [−2.0, −0.9] |
| V5a → V6 | LOBT-gap rule | −11.5 s (mostly a few rows) | [−34.9, −0.35] |
| V6 → V7 | V7c congestion fix for rows without AOBT_3 | −0.35 s (Jan −0.7, Feb −8.8) | [−0.86, −0.01] |

Honest caveats: the RMSE of these months is dominated by a few hundred rows with errors of 10³–10⁵ s, so
leaderboard differences of ±5 s between models are within the luck of those rows (V4b scored worse than V3 despite
−14.6 s in validation). The LOBT rule's form was chosen after looking at all of 2025; only its coefficients were
fitted out-of-month. V7c was the third variant tried on the same months (the first two were rejected and are
documented); the accepted one has no fitted parameter.

## Rejected routes (all documented with their protocols)

| route | where | outcome |
|---|---|---|
| OPDI (open ADS-B ground events) as off-block proxy | `outputs/experiments/v5/protocol.md` §3 | 6 % coverage, worse than the model where available |
| LightGBM linear trees; blended LightGBM gate | `v5` | ≈ 0; gain came from one row per month |
| Airline-prefix gate for LIRF flights without NM plan | `outputs/experiments/v6/protocol.md` §2 | +0.4 s |
| Own-AOBT and neighbour-anchored congestion corrections for all rows | `outputs/experiments/v7/protocol.md` | +0.1 s (overfits a February episode) |
| Learning the class of decisive rows from the leaderboard | — | would reach ≈ 230 s; not done (rules) |

## Reproduce

Hardware used: a MacBook Air (10 threads, `.venv-mac`) for the CatBoost components and a 2-CPU/6 GB cloud
container for LightGBM, validation and the final assembly. Python 3.10–3.11, `requirements.txt`
(+ `lightgbm>=4.3` for V5 onwards).

```bash
# 0. Put the 14 original Parquet files in the project root or data/raw/ (they are never modified; see "Data")
python -m pytest tests -q                       # 60 tests (feature/anti-leakage/combination/no-upload checks)

# 1. V4b components, 12 months, seeds 42/7/2025  →  models/experiments/v4b/final/*__rank.npy, candidate_v4b
bash src/v4b_run_candidate.sh                   # ≈ 2 h on 10 threads
# 2. LightGBM normal branch + V5a blend          →  models/experiments/v5/final/lgb_const__s42__rank.npy, candidate_v5a
bash src/v5a_run_candidate.sh                   # ≈ 12 min
# 3. V6 = V5a + LOBT rule, V7 = V6 + V7c (seconds; use cached predictions)
bash src/v6_run_candidate.sh && python src/v7_final.py

# 4. Final phase (4 months). Build data/final/ with final_submitting.parquet as submitting.parquet and
#    final_ranking.parquet restricted to the new months (Feb + Jun 2026) as ranking.parquet, plus links to the
#    12 training files; January/July are copied from candidate_v7.
PRC_DATA_DIR=data/final python src/final_build.py        # ≈ 22 min on 2 CPUs → submissions/candidate_final.parquet

# Validation (what produced the numbers above)
python src/v4b_protocol.py --folds LOJO LOJuO --seeds 42 7 2025 --resume          # ≈ hours, 9-month components
python src/v4b_analysis.py && python src/v4b_report.py
python src/v5_lgb.py --fold LOJO --kinds const --outer && python src/v5_lgb.py --fold LOJuO --kinds const --outer
python src/v5_eval_inner.py --fold LOJO && python src/v5_eval_inner.py --fold LOJuO && python src/v5_eval_outer.py
python src/v6_rule_check.py && python src/v6_prefix_gate.py
python src/v7_congestion.py && python src/v7b_neighbour.py && python src/v7c_noaobt.py
```

Known reproducibility limits: the final CatBoost/LightGBM models are not persisted (only their ranking
predictions, `.npy`, are); re-running the scripts reproduces the submissions byte-for-byte on the same machine
(`candidate_v6`/`v7` were rebuilt on a second machine with identical SHA-256). In the final-phase file the
February/June normal branch uses LightGBM only (the CatBoost branch could not be retrained in the time available
after the new files were published); the validated cost of dropping the blend is ≈ 1.4 s.

## Repository layout

```
├── *.parquet                      # original challenge files (read-only, not in git)
├── src/
│   ├── config.py, io_utils.py     # paths ($PRC_DATA_DIR → data/raw/ → project root), columns, SEED = 42
│   ├── validate_data.py, eda.py, run_eda.py, prepare_training.py, modeling_utils.py
│   ├── train_baseline.py, backtest.py, select_model.py, make_submission.py, make_report.py   # V1 (baselines, backtests)
│   ├── v2_*.py                    # V2: regime analysis, 8-fold protocol, congestion features, mixture
│   ├── v3_*.py                    # V3: RegimeClassifier2 with date-shift cells, candidate_v3
│   ├── v4_*.py, v4b_*.py          # V4 audits/oracles; V4b: new normal branch + multivariate gate, LOJO/LOJuO protocol
│   ├── v5_*.py, v5a_final.py      # LightGBM branch, blended gate, OPDI probe, expected-RMSE model
│   ├── v6_*.py                    # LOBT-gap rule (+ rejected airline-prefix gate)
│   ├── v7_*.py, v7b_*.py, v7c_*.py, v7_final.py   # congestion corrections (two rejected, V7c accepted)
│   └── final_build.py             # phase-2 assembly (4 months)
├── tests/                         # pytest, 60 tests
├── outputs/experiments/v2…v7, final/   # protocols (pre-registered rules), checks (JSON), reports, logs
├── outputs/{tables,figures,reports}/   # V1 EDA and backtests
├── docs/METHOD.md                 # method report for the review
├── docs/README_fases_1-3_es.md    # previous README (V1–V3, Spanish)
├── submissions/                   # candidates + JSON metadata, upload scripts, manifest (not in git)
├── models/                        # predictions and caches (not in git)
├── requirements.txt, requirements-lock.txt, LICENSE (GPL-3.0), .gitignore
```

## Data handling and anti-leakage

* The 14 original Parquet files are only read (`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` for the
  R extension type of `AIRCRAFT_OPERATOR_flt`). Nothing in `data/`, `models/`, `submissions/`, credentials or
  Parquet is tracked by git.
* `BLOCK_TIME_UTC_mvt` and the target are never read for ranking rows; `MVT_ID_mvt` is used only to align rows
  and `FLIGHT_ID_mvt` only to link arrivals and departures, never as numeric predictors.
* Regime labels are derived from the target of the training block only. Target encodings exclude the row's own
  month. Congestion and arrival-context features use only timestamps and columns that are present in the ranking
  files (including arrivals' `BLOCK_TIME`, which is visible for ARR rows).
* Fold splits are by calendar month; nothing is ever selected on the evaluation month.

## Uploads

Uploads are done by hand with the MinIO client from `submissions/upload_optimistic-dragon_*.sh`: local SHA-256
and row/schema checks, refusal to overwrite an existing remote name, remote SHA-256 verification, result written to
`submissions/upload_result_*.json`. Credentials are never stored in the repository.
