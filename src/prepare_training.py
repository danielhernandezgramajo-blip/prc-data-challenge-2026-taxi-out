"""
FASE 3 — Preparación reproducible de features para el PRC Data Challenge 2026.

Principios:
* Se trabaja en Polars *lazy* directamente sobre los Parquet originales (solo lectura).
  No se escribe una copia de los 4.2 M de filas: `load_features("train")` devuelve el
  DataFrame ya filtrado a DEP (≈2.1 M filas) en < 10 s.
* `build_features()` es UNA sola función aplicada a training y a ranking, de modo que
  ambos conjuntos tienen exactamente las mismas columnas y transformaciones.
* Ninguna feature usa el target ni `BLOCK_TIME_UTC_mvt` (oculto en ranking). Las
  features de congestión cuentan movimientos por marcas de tiempo (`MVT_TIME_UTC_mvt`),
  disponibles en ranking para todas las filas (DEP y ARR).
* `MVT_ID_mvt` se conserva únicamente para trazabilidad y para alinear la submission.
* Cadenas "NA" literales → nulo; categóricas nulas → "UNK".

Uso:
    python src/prepare_training.py        # genera outputs/tables/feature_audit.csv y comprueba train == rank
    from prepare_training import load_features
    train = load_features("train"); rank = load_features("rank")
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Literal

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from config import (  # noqa: E402
    DEP, FEATURE_CAT_COLS, FEATURE_CONGESTION_COLS, FEATURE_NUM_COLS, FORBIDDEN_FEATURES,
    ID_COL, PHASE_COL, TABLES_DIR, TARGET, UNKNOWN_CATEGORY,
)
from io_utils import scan_ranking, scan_training  # noqa: E402

# Columnas crudas necesarias (proyección para ahorrar memoria)
RAW_COLS = [
    ID_COL, PHASE_COL, TARGET, "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_RULE_mvt", "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt",
]
# Columnas de salida (orden fijo → train y rank idénticos)
META_COLS = [ID_COL, "ADEP_mvt_key", "MVT_TIME_UTC_mvt", TARGET]


def _sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float32)


def _clean_cat(c: str) -> pl.Expr:
    """'NA' literal → nulo; nulo → 'UNK'. Strings sin espacios sobrantes."""
    x = pl.col(c).cast(pl.String).str.strip_chars()
    return pl.when(x.is_null() | (x == "NA") | (x == "")).then(pl.lit(UNKNOWN_CATEGORY)).otherwise(x).alias(c)


def build_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Construye las features a partir de un LazyFrame con las columnas crudas de RAW_COLS.

    La entrada debe contener TODAS las fases (DEP y ARR) del periodo: las features de
    congestión usan las llegadas. La salida contiene solo DEP.
    """
    t = pl.col("MVT_TIME_UTC_mvt")
    lf = lf.select(RAW_COLS).with_columns(
        t.dt.truncate("1h").alias("_hour_bucket"),
        t.dt.date().alias("_day_bucket"),
        pl.col("RUNWAY_mvt").cast(pl.String).replace({"NA": None}).alias("_rwy_clean"),
    )

    # --- Congestión (solo marcas de tiempo, sin target). Se calcula sobre el periodo entero
    #     del conjunto (training completo o ranking completo): son conteos por hora/día y
    #     no dependen del fold. En ranking se calculan exactamente igual. -----------------
    dep = lf.filter(pl.col(PHASE_COL) == DEP)
    arr = lf.filter(pl.col(PHASE_COL) != DEP)
    dep_hour = dep.group_by("ADEP_mvt", "_hour_bucket").agg(pl.len().cast(pl.Float32).alias("departures_per_airport_hour"))
    dep_day = dep.group_by("ADEP_mvt", "_day_bucket").agg(pl.len().cast(pl.Float32).alias("departures_per_airport_day"))
    dep_rwy_hour = dep.group_by("ADEP_mvt", "_rwy_clean", "_hour_bucket").agg(
        pl.len().cast(pl.Float32).alias("departures_per_airport_runway_hour"))
    # Llegadas al mismo aeropuerto (ADES) en la misma hora: ocupación de pistas/calles de rodaje
    arr_hour = arr.group_by(pl.col("ADES_mvt").alias("ADEP_mvt"), "_hour_bucket").agg(
        pl.len().cast(pl.Float32).alias("arrivals_per_airport_hour"))

    out = (
        dep.join(dep_hour, on=["ADEP_mvt", "_hour_bucket"], how="left")
           .join(dep_day, on=["ADEP_mvt", "_day_bucket"], how="left")
           .join(dep_rwy_hour, on=["ADEP_mvt", "_rwy_clean", "_hour_bucket"], how="left")
           .join(arr_hour, on=["ADEP_mvt", "_hour_bucket"], how="left")
           .with_columns(pl.col("arrivals_per_airport_hour").fill_null(0.0))
    )

    # --- Calendario ----------------------------------------------------------------------
    out = out.with_columns(
        t.dt.month().cast(pl.Int8).alias("month"),
        t.dt.weekday().cast(pl.Int8).alias("day_of_week"),   # 1 = lunes … 7 = domingo
        t.dt.hour().cast(pl.Int8).alias("hour_utc"),
    ).with_columns(
        (pl.col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
        (2 * math.pi * pl.col("hour_utc") / 24).sin().cast(pl.Float32).alias("hour_sin"),
        (2 * math.pi * pl.col("hour_utc") / 24).cos().cast(pl.Float32).alias("hour_cos"),
        (2 * math.pi * (pl.col("month") - 1) / 12).sin().cast(pl.Float32).alias("month_sin"),
        (2 * math.pi * (pl.col("month") - 1) / 12).cos().cast(pl.Float32).alias("month_cos"),
    )

    # --- Operacionales (todas las marcas están disponibles en ranking; BLOCK_TIME no se usa) ---
    out = out.with_columns(
        _sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_minus_aobt3"),
        _sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_minus_eobt1"),
        _sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_minus_lobt"),
        _sec("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_minus_iobt"),
        _sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_minus_sched"),
        _sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt3_minus_sched"),
        pl.col("AOBT_3_flt").is_not_null().cast(pl.Int8).alias("has_aobt3"),
    )

    # --- Categóricas ---------------------------------------------------------------------
    out = out.with_columns([_clean_cat(c) for c in FEATURE_CAT_COLS])
    out = out.with_columns(pl.col("ADEP_mvt").alias("ADEP_mvt_key"))  # copia para métricas por aeropuerto

    return out.select(META_COLS + FEATURE_NUM_COLS + FEATURE_CAT_COLS)


def load_features(split: Literal["train", "rank"]) -> pl.DataFrame:
    """Features DEP para training (12 meses) o ranking. Mismas columnas, mismo orden."""
    lf = scan_training(RAW_COLS) if split == "train" else scan_ranking(RAW_COLS)
    df = build_features(lf).collect()
    assert_no_forbidden(df)
    return df


def assert_no_forbidden(df: pl.DataFrame) -> None:
    """Auditoría dura: ninguna columna prohibida entre las features."""
    feats = feature_columns()
    bad = [c for c in feats if c in FORBIDDEN_FEATURES]
    assert not bad, f"Columnas prohibidas entre las features: {bad}"
    assert "BLOCK_TIME_UTC_mvt" not in df.columns, "BLOCK_TIME_UTC_mvt no debe estar en el frame de features"


def feature_columns() -> list[str]:
    return FEATURE_NUM_COLS + FEATURE_CAT_COLS


# ---------------------------------------------------------------------------
# Auditoría de features
# ---------------------------------------------------------------------------
_RISK = {
    # feature: (disponibilidad en inferencia, riesgo de leakage, decisión, motivo)
    "mvt_minus_aobt3": ("sí (98.9 % train / 98.5 % rank)", "ALTO (permitido)", "incluir",
                        "AOBT_3 es el off-block real a resolución de minuto reportado por A-CDM; ≈ target pero está en ranking."),
    "aobt3_minus_sched": ("sí", "MEDIO", "incluir", "Retraso real de salida; deriva de AOBT_3."),
    "has_aobt3": ("sí", "BAJO", "incluir", "Indicador de enlace con plan de vuelo; los extremos se concentran en has_aobt3 = 0."),
    "mvt_minus_eobt1": ("sí", "BAJO", "incluir", "Planificado; fallback natural cuando falta AOBT_3 (mismo patrón de nulos)."),
    "mvt_minus_lobt": ("sí", "BAJO", "incluir", "Planificado (última EOBT)."),
    "mvt_minus_iobt": ("sí", "BAJO", "incluir", "Planificado (primera EOBT)."),
    "mvt_minus_sched": ("sí (100 %)", "BAJO", "incluir", "Programado; única delta sin nulos."),
    "departures_per_airport_hour": ("sí (conteo de MVT_TIME en ranking)", "BAJO", "incluir", "Congestión; no usa target."),
    "departures_per_airport_day": ("sí", "BAJO", "incluir", "Congestión diaria."),
    "departures_per_airport_runway_hour": ("sí", "BAJO", "incluir", "Cola en la pista; no usa target."),
    "arrivals_per_airport_hour": ("sí (filas ARR del ranking)", "BAJO", "incluir", "Ocupación por llegadas; no usa target."),
}
_CAL_NOTE = ("sí", "NULO", "incluir", "Derivada de MVT_TIME_UTC_mvt.")
_CAT_NOTE = {
    "ADEP_mvt": "Aeropuerto de salida: factor dominante.",
    "ADES_mvt": "Destino: proxy de ruta/tipo de vuelo; ~1.4 % de discrepancia con ADES_flt.",
    "RUNWAY_mvt": "Pista; 'NA' → UNK.",
    "STAND_mvt": "Stand: alta cardinalidad (≈2.7 k); 0.46 % de filas de ranking con stand no visto → UNK/hash.",
    "AIRCRAFT_TYPE_mvt": "Tipo de aeronave (redundante con AIRCRAFT_TYPE_flt: 99.7 %).",
    "MARKET_SEGMENT_flt": "Segmento de mercado.",
    "FLIGHT_TYPE_flt": "Tipo de vuelo.",
    "WK_TBL_CAT_flt": "Categoría de estela.",
    "AIRCRAFT_OPERATOR_flt": "Operador (hash); 1.6 % de filas de ranking con operador no visto.",
    "FLIGHT_RULE_mvt": "Reglas de vuelo.",
}
_EXCLUDED = [
    ("TAXITIME_SEC_mvt", "numérica (target)", "no (nulo en ranking-DEP)", "TARGET", "excluir", "Es el objetivo."),
    ("BLOCK_TIME_UTC_mvt", "temporal", "no (nulo en ranking-DEP)", "PROHIBIDO", "excluir", "Target = MVT_TIME − BLOCK_TIME exactamente."),
    ("MVT_ID_mvt", "identificador", "sí", "MEDIO (correlaciona con el tiempo)", "excluir (solo trazabilidad)", "Identificador."),
    ("FLIGHT_ID_mvt", "identificador", "sí", "MEDIO", "excluir", "Identificador creciente en el tiempo."),
    ("FLIGHT_mvt / CALLSIGN_flt", "categórica", "sí", "BAJO", "excluir (por ahora)", "Cardinalidad muy alta; redundante con operador+ruta. Candidata a target encoding en fases posteriores."),
    ("ADEP_flt / AIRCRAFT_TYPE_flt / FLIGHT_RULE_flt / ADES_FILED_flt", "categórica", "sí", "BAJO", "excluir", "Redundantes (> 99.6 %) con sus homólogas _mvt / ADES_flt."),
    ("ADES_flt", "categórica", "sí", "BAJO", "excluir", "98.6 % igual a ADES_mvt; se usa ADES_mvt."),
    ("ARVT_1_flt / ARVT_3_flt", "temporal", "sí", "BAJO", "excluir", "Posteriores al despegue; solo informan de la duración del vuelo."),
    ("PHASE_mvt", "categórica", "sí", "NULO", "excluir", "Constante (= DEP) tras el filtro."),
    ("MVT_TIME_UTC_mvt y demás marcas absolutas", "temporal", "sí", "BAJO", "excluir como valor absoluto", "Solo se usan a través de deltas y calendario (evita extrapolar la tendencia temporal a 2026)."),
]


def feature_audit(train: pl.DataFrame, rank: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for c in feature_columns():
        kind = "categórica" if c in FEATURE_CAT_COLS else ("congestión" if c in FEATURE_CONGESTION_COLS else
                                                         ("calendario" if c in ("month", "day_of_week", "hour_utc", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos") else "operacional"))
        if c in FEATURE_CAT_COLS:
            avail, risk, dec, why = ("sí", "BAJO", "incluir", _CAT_NOTE[c])
            null_tr = 100 * (train[c] == UNKNOWN_CATEGORY).mean(); null_rk = 100 * (rank[c] == UNKNOWN_CATEGORY).mean()
        else:
            avail, risk, dec, why = _RISK.get(c, _CAL_NOTE)
            null_tr = 100 * train[c].null_count() / train.height; null_rk = 100 * rank[c].null_count() / rank.height
        rows.append({"variable": c, "tipo": kind, "pct_nulos_train": round(null_tr, 3), "pct_nulos_rank": round(null_rk, 3),
                     "cardinalidad_train": train[c].n_unique(), "cardinalidad_rank": rank[c].n_unique(),
                     "disponible_en_inferencia": avail, "riesgo_leakage": risk, "decision": dec, "motivo": why})
    for v, kind, avail, risk, dec, why in _EXCLUDED:
        rows.append({"variable": v, "tipo": kind, "pct_nulos_train": None, "pct_nulos_rank": None, "cardinalidad_train": None,
                     "cardinalidad_rank": None, "disponible_en_inferencia": avail, "riesgo_leakage": risk, "decision": dec, "motivo": why})
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(TABLES_DIR / "feature_audit.csv")
    return df


def check_same_schema(train: pl.DataFrame, rank: pl.DataFrame) -> None:
    assert train.columns == rank.columns, f"Columnas distintas:\n{train.columns}\n{rank.columns}"
    assert [str(t) for t in train.dtypes] == [str(t) for t in rank.dtypes], "Tipos distintos entre train y rank"


def main() -> None:
    import time
    t0 = time.time()
    train = load_features("train")
    rank = load_features("rank")
    check_same_schema(train, rank)
    print(f"train DEP {train.height:,} × {train.width} | rank DEP {rank.height:,} × {rank.width} | {time.time()-t0:.1f}s")
    print(f"features: {len(feature_columns())} ({len(FEATURE_NUM_COLS)} numéricas + {len(FEATURE_CAT_COLS)} categóricas)")
    audit = feature_audit(train, rank)
    print(audit.select("variable", "tipo", "pct_nulos_train", "pct_nulos_rank", "cardinalidad_train", "decision"))
    assert rank[TARGET].null_count() == rank.height, "El ranking no debe tener target"
    print("Auditoría escrita en outputs/tables/feature_audit.csv")


if __name__ == "__main__":
    main()
