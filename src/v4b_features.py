"""
V4b — Tabla de features para training o ranking (mismo código), sin target ni BLOCK_TIME de las salidas.

    python src/v4b_features.py --split train      # → models/experiments/v4b/features_train.parquet (fuera de git)

Base: V1 + congestión V2 + `days_after_sched` (load_features_v3(split, surface=True)). Las relativas de V2 se ajustan por fold
(RelativeCongestion) en v4b_protocol. Nuevas (ver protocol.md §3):
  plan         eobt1_minus_sched, lobt_minus_sched, iobt_minus_sched, aobt3_minus_eobt1, eobt1_minus_lobt, lobt_minus_iobt
  resolución   aobt3_sec0, mvt_sec0
  escala       min_since_prev_arr, prev_arr_delay_min, prev_arr_taxi_in, prev_arr_copy   (llegada previa, mismo aeropuerto+stand)
  contexto     arr_copy_rate_pm60, arr_copy_rate_pm180, arr_n_pm60   (llegadas del aeropuerto con BLOCK = SCHED)
  frecuencia   flight_n_month
  categóricas  FLIGHT_mvt, CALLSIGN_flt, apt_stand, apt_runway, apt_operator, route
Las columnas BLOCK/TAXITIME solo se leen para las filas ARR (visibles en ranking); de las DEP se descartan al leer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import DEP, FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, MODELS_DIR, PHASE_COL, UNKNOWN_CATEGORY  # noqa: E402
from io_utils import scan_ranking, scan_training  # noqa: E402

EXP_DIR = MODELS_DIR / "experiments" / "v4b"
COPY_TOL = 10.0
V4B_NEW_NUM = ["eobt1_minus_sched", "lobt_minus_sched", "iobt_minus_sched", "aobt3_minus_eobt1", "eobt1_minus_lobt",
               "lobt_minus_iobt", "aobt3_sec0", "mvt_sec0", "min_since_prev_arr", "prev_arr_delay_min", "prev_arr_taxi_in",
               "prev_arr_copy", "arr_copy_rate_pm60", "arr_copy_rate_pm180", "arr_n_pm60", "flight_n_month"]
V4B_NEW_CAT = ["FLIGHT_mvt", "CALLSIGN_flt", "apt_stand", "apt_runway", "apt_operator", "route"]
DEP_RAW = [ID_COL, PHASE_COL, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt",
           "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt", "ADEP_mvt"]
ARR_RAW = [PHASE_COL, "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def num_cols() -> list[str]:
    from v2_features import V2_NUM_COLS, V2_REL_COLS
    from v3_protocol import DATE_SHIFT_COL
    return FEATURE_NUM_COLS + V2_NUM_COLS + V2_REL_COLS + [DATE_SHIFT_COL] + V4B_NEW_NUM


def cat_cols() -> list[str]:
    from v2_features import V2_CAT_COLS
    return FEATURE_CAT_COLS + V2_CAT_COLS + V4B_NEW_CAT


def _scan(split: str, cols: list[str]) -> pl.LazyFrame:
    return scan_training(cols) if split == "train" else scan_ranking(cols)


def _clean(c: str, alias: str | None = None) -> pl.Expr:
    x = pl.col(c).cast(pl.String).str.strip_chars()
    return pl.when(x.is_null() | (x == "NA") | (x == "")).then(pl.lit(UNKNOWN_CATEGORY)).otherwise(x).alias(alias or c)


def _sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float32)


def dep_frame(split: str) -> pl.DataFrame:
    """Columnas visibles de las salidas (nunca BLOCK ni TAXITIME)."""
    d = _scan(split, DEP_RAW).filter(pl.col(PHASE_COL) == DEP).collect()
    assert "BLOCK_TIME_UTC_mvt" not in d.columns and "TAXITIME_SEC_mvt" not in d.columns
    return d


def arr_frame(split: str) -> pl.DataFrame:
    """Llegadas: aeropuerto, stand, aterrizaje, on-block, programado, taxi-in (visibles en ranking para ARR)."""
    a = _scan(split, ARR_RAW).filter(pl.col(PHASE_COL) != DEP).collect()
    return a.select(pl.col("ADES_mvt").cast(pl.String).alias("apt"), _clean("STAND_mvt", "stand"),
                    pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_arr"),
                    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60).cast(pl.Float32).alias("prev_arr_delay_min"),
                    pl.col("TAXITIME_SEC_mvt").cast(pl.Float32).alias("prev_arr_taxi_in"),
                    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= COPY_TOL)
                    .cast(pl.Float32).alias("prev_arr_copy"))


def window_rate(t_q: np.ndarray, t_ev: np.ndarray, flag: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Para cada consulta: nº de eventos con t ∈ [t_q − w, t_q + w] y fracción de ellos con flag (nan si no hay).
    t_ev debe estar ordenado; flag alineado con t_ev."""
    cs = np.concatenate([[0.0], np.cumsum(flag, dtype=np.float64)])
    lo = np.searchsorted(t_ev, t_q - w, side="left"); hi = np.searchsorted(t_ev, t_q + w, side="right")
    n = (hi - lo).astype(np.float64); k = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return n, rate


def new_features(split: str) -> pl.DataFrame:
    d = dep_frame(split)
    a = arr_frame(split)
    t_mvt = pl.col("MVT_TIME_UTC_mvt")
    f = d.select(
        ID_COL,
        pl.col("ADEP_mvt").cast(pl.String).alias("apt"), _clean("STAND_mvt", "stand"),
        t_mvt.dt.epoch("s").cast(pl.Float64).alias("t_mvt"),
        _sec("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("eobt1_minus_sched"),
        _sec("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("lobt_minus_sched"),
        _sec("IOBT_flt", "SCHED_TIME_UTC_mvt").alias("iobt_minus_sched"),
        _sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt3_minus_eobt1"),
        _sec("EOBT_1_flt", "LOBT_flt").alias("eobt1_minus_lobt"),
        _sec("LOBT_flt", "IOBT_flt").alias("lobt_minus_iobt"),
        pl.when(pl.col("AOBT_3_flt").is_null()).then(-1).otherwise((pl.col("AOBT_3_flt").dt.second() == 0).cast(pl.Int8))
          .cast(pl.Float32).alias("aobt3_sec0"),
        (t_mvt.dt.second() == 0).cast(pl.Float32).alias("mvt_sec0"),
        _clean("FLIGHT_mvt"), _clean("CALLSIGN_flt"),
        t_mvt.dt.strftime("%Y-%m").alias("_ym"),
    )
    f = f.with_columns(pl.len().over("FLIGHT_mvt", "_ym").cast(pl.Float32).alias("flight_n_month")).drop("_ym")
    # llegada previa al mismo aeropuerto+stand antes del despegue (≤ 24 h), como en exp_capacity
    prev = (f.select(ID_COL, "apt", "stand", "t_mvt").sort("t_mvt")
              .join_asof(a.sort("t_arr"), left_on="t_mvt", right_on="t_arr", by=["apt", "stand"], strategy="backward",
                         tolerance=24 * 3600.0, check_sortedness=False)
              .with_columns(((pl.col("t_mvt") - pl.col("t_arr")) / 60).cast(pl.Float32).alias("min_since_prev_arr"))
              .select(ID_COL, "min_since_prev_arr", "prev_arr_delay_min", "prev_arr_taxi_in", "prev_arr_copy"))
    f = f.join(prev, on=ID_COL, how="left")
    # tasa de artefacto (BLOCK = SCHED) en las llegadas del aeropuerto en ±60 / ±180 min
    parts = []
    arr_by = {k[0]: s.sort("t_arr") for k, s in a.group_by("apt")}
    for (apt,), sub in f.select(ID_COL, "apt", "t_mvt").group_by("apt"):
        sa = arr_by.get(apt)
        t = sub["t_mvt"].to_numpy()
        if sa is None or sa.height == 0:
            nan = np.full(t.size, np.nan, dtype=np.float32)
            parts.append(sub.select(ID_COL).with_columns(pl.Series("arr_n_pm60", np.zeros(t.size, dtype=np.float32)),
                                                         pl.Series("arr_copy_rate_pm60", nan), pl.Series("arr_copy_rate_pm180", nan)))
            continue
        te = sa["t_arr"].to_numpy(); fl = sa["prev_arr_copy"].to_numpy().astype(np.float64)
        n60, r60 = window_rate(t, te, fl, 3600.0); _, r180 = window_rate(t, te, fl, 3 * 3600.0)
        parts.append(sub.select(ID_COL).with_columns(pl.Series("arr_n_pm60", n60.astype(np.float32)),
                                                     pl.Series("arr_copy_rate_pm60", r60.astype(np.float32)),
                                                     pl.Series("arr_copy_rate_pm180", r180.astype(np.float32))))
    f = f.join(pl.concat(parts), on=ID_COL, how="left")
    return f.drop("apt", "stand", "t_mvt")


def build_v4b(split: str) -> pl.DataFrame:
    from v3_protocol import load_features_v3
    base = load_features_v3(split, surface=True)
    new = new_features(split)
    out = base.join(new, on=ID_COL, how="left")
    assert out.height == base.height
    out = out.with_columns(
        (pl.col("ADEP_mvt") + "|" + pl.col("STAND_mvt")).alias("apt_stand"),
        (pl.col("ADEP_mvt") + "|" + pl.col("RUNWAY_mvt")).alias("apt_runway"),
        (pl.col("ADEP_mvt") + "|" + pl.col("AIRCRAFT_OPERATOR_flt")).alias("apt_operator"),
        (pl.col("ADEP_mvt") + ">" + pl.col("ADES_mvt")).alias("route"),
    )
    missing = [c for c in num_cols() + cat_cols() if c not in out.columns and c not in ("rel_dep_pm30", "rel_dep_hour")]
    assert not missing, missing
    assert "BLOCK_TIME_UTC_mvt" not in out.columns
    return out


def features_path(split: str) -> Path:
    return EXP_DIR / f"features_{split}.parquet"


def load_v4b(split: str) -> pl.DataFrame:
    p = features_path(split)
    if not p.exists():
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        build_v4b(split).write_parquet(p)
    return pl.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "rank"], default="train")
    args = ap.parse_args()
    t0 = time.time()
    df = build_v4b(args.split)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(features_path(args.split))
    print(f"{args.split}: {df.shape} en {time.time() - t0:.0f}s → {features_path(args.split)}")
    new = [c for c in V4B_NEW_NUM]
    print(df.select([(pl.col(c).is_null().mean() * 100).round(2).alias(c) for c in new]).transpose(include_header=True,
          column_names=["pct_null"]))


if __name__ == "__main__":
    main()
