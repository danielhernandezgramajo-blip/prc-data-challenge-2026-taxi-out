"""
Configuración central del proyecto PRC Data Challenge 2026 (taxi-out time).

Todas las rutas son relativas a la raíz del proyecto (la carpeta que contiene
`src/`). Los Parquet originales NO se mueven: se buscan primero en `data/raw/`
y, si no están ahí, en la raíz del proyecto (ubicación actual). Se puede
sobreescribir con la variable de entorno `PRC_DATA_DIR`.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def resolve_data_dir() -> Path:
    """Devuelve la carpeta que contiene los 14 Parquet originales."""
    env = os.environ.get("PRC_DATA_DIR")
    candidates = [Path(env)] if env else []
    candidates += [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT]
    for c in candidates:
        if (c / "submitting.parquet").exists():
            return c
    raise FileNotFoundError(
        "No se encontró submitting.parquet. Coloca los Parquet en data/raw/ "
        "o en la raíz del proyecto, o define PRC_DATA_DIR."
    )


DATA_DIR = resolve_data_dir()

# ---------------------------------------------------------------------------
# Archivos esperados
# ---------------------------------------------------------------------------
TRAINING_MONTHS = [
    ("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01"),
    ("2025-03-01", "2025-04-01"), ("2025-04-01", "2025-05-01"),
    ("2025-05-01", "2025-06-01"), ("2025-06-01", "2025-07-01"),
    ("2025-07-01", "2025-08-01"), ("2025-08-01", "2025-09-01"),
    ("2025-09-01", "2025-10-01"), ("2025-10-01", "2025-11-01"),
    ("2025-11-01", "2025-12-01"), ("2025-12-01", "2026-01-01"),
]
TRAINING_FILES = [f"training_{a}_{b}.parquet" for a, b in TRAINING_MONTHS]
RANKING_FILE = "ranking.parquet"
SUBMITTING_FILE = "submitting.parquet"

EXPECTED_TRAINING_ROWS = 4_167_797
EXPECTED_RANKING_ROWS = 689_534
EXPECTED_SUBMITTING_ROWS = 344_841
EXPECTED_N_COLS = 30
EXPECTED_SUBMITTING_COLS = 2

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
ID_COL = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PHASE_COL = "PHASE_mvt"
DEP = "DEP"

# Clasificación de las 30 variables (ver diccionario en outputs/tables/data_dictionary.csv)
ID_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt"]
TIME_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]
CAT_COLS = [
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "CALLSIGN_flt", "ADEP_flt",
    "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
]
NUM_COLS = ["TAXITIME_SEC_mvt"]

# Variables que en `ranking.parquet` son nulas para DEP (no utilizables como predictores)
HIDDEN_IN_RANKING_DEP = ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# FASES 3–5 — modelado
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"

# Features categóricas usadas por los modelos (nulos y "NA" → "UNK")
FEATURE_CAT_COLS = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
    "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_RULE_mvt",
]
FEATURE_CALENDAR_COLS = ["month", "day_of_week", "hour_utc", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"]
FEATURE_OPERATIONAL_COLS = ["mvt_minus_aobt3", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt",
                            "mvt_minus_sched", "aobt3_minus_sched", "has_aobt3"]
FEATURE_CONGESTION_COLS = ["departures_per_airport_hour", "departures_per_airport_day",
                           "departures_per_airport_runway_hour", "arrivals_per_airport_hour"]
FEATURE_NUM_COLS = FEATURE_CALENDAR_COLS + FEATURE_OPERATIONAL_COLS + FEATURE_CONGESTION_COLS
UNKNOWN_CATEGORY = "UNK"

# Variables que NUNCA pueden entrar como predictor (auditadas en prepare_training.py)
FORBIDDEN_FEATURES = ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "source_file"]

# Baseline jerárquica: mínimo de observaciones por grupo para aceptar su mediana
MIN_GROUP_N = 30
HIER_LEVELS = [
    ["ADEP_mvt", "RUNWAY_mvt", "hour_utc", "AIRCRAFT_TYPE_mvt"],
    ["ADEP_mvt", "RUNWAY_mvt", "hour_utc"],
    ["ADEP_mvt", "RUNWAY_mvt"],
    ["ADEP_mvt"],
]

# Backtests temporales (mes de inicio del training, mes final del training, mes de validación) — todo 2025
FOLDS = {
    "F1_jul": (1, 6, 7),
    "F2_nov": (1, 10, 11),
    "F3_dec": (1, 11, 12),
}

# Estrategias de tratamiento del target SOLO en entrenamiento (la validación siempre usa el target original)
OUTLIER_STRATEGIES = ["original", "drop_nonpositive", "drop_gt3h", "winsor_p01_p999"]
OUTLIER_HI_SEC = 3 * 3600

# CatBoost: configuración moderada (VM de 3 GB / 4 CPU)
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "max_ctr_complexity": 1,
    "one_hot_max_size": 12,
    "random_seed": SEED,
    "thread_count": 4,
    "od_type": "Iter",
    "od_wait": 80,
    "verbose": 200,
    "allow_writing_files": False,
}
EARLY_STOP_DAYS = 14  # últimos N días del bloque de entrenamiento → conjunto de early stopping (temporal, no aleatorio)
