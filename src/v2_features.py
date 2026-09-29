"""
V2 — Features de congestión temporal (familia A).

Se construyen SOLO a partir de marcas de tiempo y claves disponibles en training y en ranking
(`MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, `ADEP_mvt`/`ADES_mvt`, `RUNWAY_mvt`, `PHASE_mvt`).
Nunca se lee `TAXITIME_SEC_mvt` ni `BLOCK_TIME_UTC_mvt` (ni siquiera se proyectan).

Features (todas excluyen la propia fila cuando es un conteo):
  dep_pm15, dep_pm30           salidas del mismo aeropuerto en ±15 / ±30 min
  rwy_prev10, rwy_prev15, rwy_prev30   salidas de la misma pista en los 10/15/30 min ANTERIORES
  arr_pm15, arr_pm30           llegadas al aeropuerto en ±15 / ±30 min
  net_pm15, net_pm30           presión neta = salidas − llegadas por ventana
  min_since_prev_rwy_dep       minutos desde la salida anterior de la misma pista (MVT_TIME)
  min_to_next_rwy_sched        minutos hasta la siguiente salida PROGRAMADA de la misma pista (SCHED_TIME)
  apt_hour_dow                 interacción categórica aeropuerto × hora × día de semana
  rel_dep_pm30, rel_dep_hour   congestión relativa: dep_pm30 / mediana histórica(aeropuerto, hora) y
                               departures_per_airport_hour / mediana histórica(aeropuerto, hora)

La "mediana histórica" es una agregación SIN target (conteos de salidas) calculada exclusivamente con
las filas anteriores al corte del fold (`fit_relative_baseline(train_rows)`), y para ranking con todo 2025.

Ventanas: los conteos se calculan dentro del conjunto que se está procesando (training completo o ranking
completo), igual que las features de congestión por hora de V1; en los bordes de mes del ranking las
ventanas quedan truncadas exactamente igual que en los bordes de los archivos de training.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import DEP, ID_COL, PHASE_COL, UNKNOWN_CATEGORY  # noqa: E402
from io_utils import scan_ranking, scan_training  # noqa: E402
from prepare_training import build_features, RAW_COLS  # noqa: E402

V2_NUM_COLS = ["dep_pm15", "dep_pm30", "rwy_prev10", "rwy_prev15", "rwy_prev30", "arr_pm15", "arr_pm30",
               "net_pm15", "net_pm30", "min_since_prev_rwy_dep", "min_to_next_rwy_sched"]
V2_REL_COLS = ["rel_dep_pm30", "rel_dep_hour"]
V2_CAT_COLS = ["apt_hour_dow"]
FORBIDDEN_RAW = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"}
# Columnas crudas que lee este módulo (auditables): sin target ni BLOCK_TIME
V2_RAW_COLS = [ID_COL, PHASE_COL, "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
assert not FORBIDDEN_RAW & set(V2_RAW_COLS)


def _window_counts(t_query: np.ndarray, t_events: np.ndarray, lo: float, hi: float, exclude_self: bool) -> np.ndarray:
    """Nº de eventos con t in [t_query+lo, t_query+hi]. t_events debe estar ordenado. Excluye la fila
    propia si exclude_self (los eventos son el mismo conjunto que las consultas)."""
    left = np.searchsorted(t_events, t_query + lo, side="left")
    right = np.searchsorted(t_events, t_query + hi, side="right")
    c = right - left
    if exclude_self:
        c = c - 1
    return c.astype(np.float32)


def _per_group(df: pl.DataFrame, keys: list[str], fn) -> pl.DataFrame:
    """Aplica fn(sub: DataFrame ordenado por MVT_TIME) → DataFrame[ID_COL, feats] por grupo y concatena."""
    parts = []
    for _, sub in df.group_by(keys, maintain_order=True):
        parts.append(fn(sub.sort("MVT_TIME_UTC_mvt")))
    return pl.concat(parts, how="vertical_relaxed")


def congestion_features(raw: pl.DataFrame) -> pl.DataFrame:
    """raw: columnas V2_RAW_COLS de TODAS las fases (DEP y ARR). Devuelve [MVT_ID_mvt] + V2_NUM_COLS + V2_CAT_COLS
    para las filas DEP."""
    assert not FORBIDDEN_RAW & set(raw.columns), "columnas prohibidas en la entrada de congestion_features"
    raw = raw.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)).alias("_t"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)).alias("_ts"),
        pl.col("RUNWAY_mvt").cast(pl.String).replace({"NA": None}).fill_null(UNKNOWN_CATEGORY).alias("_rwy"),
    )
    dep = raw.filter(pl.col(PHASE_COL) == DEP)
    arr = raw.filter(pl.col(PHASE_COL) != DEP).select(pl.col("ADES_mvt").alias("ADEP_mvt"), "_t")

    # --- por aeropuerto: salidas ±15/±30 (sin la propia fila), llegadas ±15/±30 ---
    arr_by_apt = {k[0] if isinstance(k, tuple) else k: np.sort(s["_t"].to_numpy()) for k, s in arr.group_by("ADEP_mvt")}

    def apt_fn(sub: pl.DataFrame) -> pl.DataFrame:
        t = sub["_t"].to_numpy()
        apt = sub["ADEP_mvt"][0]
        ta = arr_by_apt.get(apt, np.empty(0))
        return pl.DataFrame({
            ID_COL: sub[ID_COL],
            "dep_pm15": _window_counts(t, t, -900, 900, True), "dep_pm30": _window_counts(t, t, -1800, 1800, True),
            "arr_pm15": _window_counts(t, ta, -900, 900, False), "arr_pm30": _window_counts(t, ta, -1800, 1800, False),
        })
    f_apt = _per_group(dep, ["ADEP_mvt"], apt_fn)

    # --- por aeropuerto × pista: salidas previas 10/15/30 min, minutos desde la anterior salida, minutos
    #     hasta la siguiente salida programada (según SCHED_TIME de la misma pista) ---
    def rwy_fn(sub: pl.DataFrame) -> pl.DataFrame:
        t = sub["_t"].to_numpy()
        prev = np.empty(t.size, dtype=np.float32); prev[0] = np.nan
        if t.size > 1:
            prev[1:] = (t[1:] - t[:-1]) / 60.0
        ts = sub["_ts"].to_numpy()
        ts_sorted = np.sort(ts[~np.isnan(ts)])
        # siguiente salida programada estrictamente posterior a la hora programada de la fila
        nxt = np.full(t.size, np.nan, dtype=np.float32)
        ok = ~np.isnan(ts)
        idx = np.searchsorted(ts_sorted, ts[ok], side="right")
        has = idx < ts_sorted.size
        vals = np.full(ok.sum(), np.nan, dtype=np.float32)
        vals[has] = (ts_sorted[idx[has]] - ts[ok][has]) / 60.0
        nxt[ok] = vals
        return pl.DataFrame({
            ID_COL: sub[ID_COL],
            "rwy_prev10": _window_counts(t, t, -600, -1e-6, False), "rwy_prev15": _window_counts(t, t, -900, -1e-6, False),
            "rwy_prev30": _window_counts(t, t, -1800, -1e-6, False),
            "min_since_prev_rwy_dep": prev, "min_to_next_rwy_sched": nxt,
        })
    f_rwy = _per_group(dep, ["ADEP_mvt", "_rwy"], rwy_fn)

    out = (dep.select(ID_COL, "ADEP_mvt", "MVT_TIME_UTC_mvt")
              .join(f_apt, on=ID_COL, how="left").join(f_rwy, on=ID_COL, how="left")
              .with_columns((pl.col("dep_pm15") - pl.col("arr_pm15")).alias("net_pm15"),
                            (pl.col("dep_pm30") - pl.col("arr_pm30")).alias("net_pm30"),
                            (pl.col("ADEP_mvt") + "_h" + pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.String)
                             + "_d" + pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.String)).alias("apt_hour_dow")))
    return out.select([ID_COL] + V2_NUM_COLS + V2_CAT_COLS)


def load_features_v2(split: str) -> pl.DataFrame:
    """Features V1 (build_features) + features de congestión V2, para training (DEP) o ranking (DEP)."""
    lf = scan_training(RAW_COLS) if split == "train" else scan_ranking(RAW_COLS)
    base = build_features(lf).collect()
    raw_lf = scan_training(V2_RAW_COLS) if split == "train" else scan_ranking(V2_RAW_COLS)
    cong = congestion_features(raw_lf.collect())
    out = base.join(cong, on=ID_COL, how="left")
    assert out.height == base.height
    assert not FORBIDDEN_RAW & (set(V2_NUM_COLS) | set(V2_CAT_COLS))
    return out


class RelativeCongestion:
    """Mediana histórica de dep_pm30 y de departures_per_airport_hour por (aeropuerto, hora), ajustada SOLO con las
    filas anteriores al corte del fold (sin target). rel_* = valor / mediana histórica (1.0 si no hay historia)."""

    def fit(self, train_rows: pl.DataFrame) -> "RelativeCongestion":
        self.table = (train_rows.group_by("ADEP_mvt", "hour_utc")
                      .agg(pl.col("dep_pm30").median().alias("_m30"), pl.col("departures_per_airport_hour").median().alias("_mh")))
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        d = df.join(self.table, on=["ADEP_mvt", "hour_utc"], how="left")
        return d.with_columns(
            pl.when(pl.col("_m30") > 0).then(pl.col("dep_pm30") / pl.col("_m30")).otherwise(1.0).cast(pl.Float32).alias("rel_dep_pm30"),
            pl.when(pl.col("_mh") > 0).then(pl.col("departures_per_airport_hour") / pl.col("_mh")).otherwise(1.0).cast(pl.Float32).alias("rel_dep_hour"),
        ).drop("_m30", "_mh")
