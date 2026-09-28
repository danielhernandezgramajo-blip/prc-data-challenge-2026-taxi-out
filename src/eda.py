"""
FASE 2 — Funciones de análisis exploratorio (EDA) para el PRC Data Challenge 2026.

Diseño:
* Todo el cálculo pesado vive aquí (Polars, lazy, proyección de columnas) para que
  el notebook `notebooks/01_eda.ipynb` sea ligero, reproducible y ejecutable de
  principio a fin en una máquina con ~3–4 GB de RAM.
* Cada función escribe sus tablas en `outputs/tables/` y sus figuras en
  `outputs/figures/` y devuelve los DataFrames para mostrarlos en el notebook.
* Solo se analizan movimientos DEP (PHASE_mvt == "DEP") salvo cuando se usa ARR
  como contexto (taxi-in disponible en ranking).
* NO se eliminan outliers: se cuantifican y se evalúan tratamientos en fases
  posteriores.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import (  # noqa: E402
    CAT_COLS, DATA_DIR, DEP, FIGURES_DIR, HIDDEN_IN_RANKING_DEP, ID_COL, ID_COLS,
    NUM_COLS, PHASE_COL, RANKING_FILE, SEED, TABLES_DIR, TARGET, TIME_COLS,
    TRAINING_FILES,
)

np.random.seed(SEED)

# Paleta (dataviz skill, modo claro): azul = training, naranja = ranking, aqua = tercera serie
C_TRAIN, C_RANK, C_3, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Diccionario de variables (interpretación a partir de nombres, PRC Data Challenge
# y Network Manager / ATFM; sufijo _mvt = registro de movimiento de aeropuerto,
# sufijo _flt = plan de vuelo de Network Manager)
# ---------------------------------------------------------------------------
DICTIONARY: dict[str, tuple[str, str]] = {
    "MVT_ID_mvt": ("identificador", "ID único del movimiento (fila). Clave de la submission. NO es predictor."),
    "FLIGHT_ID_mvt": ("identificador", "ID del vuelo en el sistema de movimientos (nulo cuando no se pudo enlazar)."),
    "FLIGHT_mvt": ("categórica", "Número de vuelo comercial (p. ej. LH982)."),
    "FLIGHT_RULE_mvt": ("categórica", "Reglas de vuelo del movimiento (I = IFR, V = VFR…)."),
    "ADEP_mvt": ("categórica", "Aeródromo de salida (ICAO). Para DEP es el aeropuerto donde ocurre el taxi-out."),
    "ADES_mvt": ("categórica", "Aeródromo de destino (ICAO)."),
    "PHASE_mvt": ("categórica", "Fase del movimiento: DEP (salida) o ARR (llegada). Solo DEP se predice."),
    "MVT_TIME_UTC_mvt": ("temporal", "Hora del movimiento: despegue real (ATOT) para DEP, aterrizaje (ALDT) para ARR. Disponible en ranking."),
    "BLOCK_TIME_UTC_mvt": ("temporal", "Hora real de bloques: AOBT (off-block) para DEP, AIBT (in-block) para ARR. OCULTA en ranking-DEP → define el target."),
    "SCHED_TIME_UTC_mvt": ("temporal", "Hora programada del movimiento (STD para DEP)."),
    "AIRCRAFT_TYPE_mvt": ("categórica", "Tipo de aeronave ICAO según el aeropuerto."),
    "RUNWAY_mvt": ("categórica", "Pista utilizada (contiene cadenas 'NA' literales)."),
    "STAND_mvt": ("categórica", "Stand/puesto de estacionamiento de origen."),
    "TAXITIME_SEC_mvt": ("numérica (target)", "Taxi-out en segundos = MVT_TIME − BLOCK_TIME. Target (nulo en ranking-DEP)."),
    "LOBT_flt": ("temporal", "Last Off-Block Time del plan de vuelo (última EOBT válida)."),
    "CALLSIGN_flt": ("categórica", "Indicativo ATC del plan de vuelo."),
    "ADEP_flt": ("categórica", "Aeródromo de salida según plan de vuelo (redundante con ADEP_mvt)."),
    "ADES_flt": ("categórica", "Aeródromo de destino real según plan de vuelo."),
    "ADES_FILED_flt": ("categórica", "Aeródromo de destino presentado originalmente."),
    "MARKET_SEGMENT_flt": ("categórica", "Segmento de mercado (Mainline, Low-cost, Regional, Business, Cargo…)."),
    "IOBT_flt": ("temporal", "Initial Off-Block Time (primera EOBT presentada)."),
    "FLIGHT_RULE_flt": ("categórica", "Reglas de vuelo del plan de vuelo."),
    "FLIGHT_TYPE_flt": ("categórica", "Tipo de vuelo (S = scheduled, N = non-scheduled, G = general…)."),
    "AIRCRAFT_TYPE_flt": ("categórica", "Tipo de aeronave ICAO según plan de vuelo."),
    "WK_TBL_CAT_flt": ("categórica", "Categoría de estela turbulenta (L/M/H/J)."),
    "AIRCRAFT_OPERATOR_flt": ("categórica", "Operador (hash SHA-256 anonimizado)."),
    "EOBT_1_flt": ("temporal", "EOBT del primer plan de vuelo (FPL) recibido."),
    "ARVT_1_flt": ("temporal", "Hora de llegada estimada/real del primer mensaje (ARVT). Posterior al despegue."),
    "AOBT_3_flt": ("temporal", "Actual Off-Block Time reportada vía DPI/A-CDM (resolución de minuto). Disponible en ranking: cuasi-target."),
    "ARVT_3_flt": ("temporal", "Hora de llegada real (ARVT) del último mensaje. Posterior al despegue."),
}

# Columnas para el análisis principal (se evita cargar todo para ahorrar memoria)
BASE_COLS = [
    ID_COL, PHASE_COL, TARGET, "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt",
    "WK_TBL_CAT_flt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt",
]


def _save(df: pl.DataFrame, name: str) -> pl.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(TABLES_DIR / f"{name}.csv")
    return df


def _fig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()


def _scan_train(cols: list[str] | None = None) -> pl.LazyFrame:
    lfs = [pl.scan_parquet(DATA_DIR / f) for f in TRAINING_FILES]
    if cols:
        lfs = [lf.select(cols) for lf in lfs]
    return pl.concat(lfs)


def _scan_rank(cols: list[str] | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(DATA_DIR / RANKING_FILE)
    return lf.select(cols) if cols else lf


def load_dep(cols: list[str] = BASE_COLS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training-DEP y ranking-DEP con columnas base + variables de calendario."""
    def enrich(lf: pl.LazyFrame) -> pl.LazyFrame:
        t = pl.col("MVT_TIME_UTC_mvt")
        return lf.filter(pl.col(PHASE_COL) == DEP).with_columns(
            t.dt.month().alias("month"), t.dt.weekday().alias("dow"), t.dt.hour().alias("hour"),
            (t.dt.weekday() >= 6).alias("is_weekend"),
            # Cadenas "NA" literales → nulo (solo en memoria; los Parquet no se tocan)
            pl.when(pl.col("RUNWAY_mvt") == "NA").then(None).otherwise(pl.col("RUNWAY_mvt")).alias("RUNWAY_mvt"),
        )
    tr = enrich(_scan_train(cols)).collect()
    rk = enrich(_scan_rank(cols)).collect()
    return tr, rk


# ---------------------------------------------------------------------------
# 1. Diccionario + clasificación + nulos + cardinalidad
# ---------------------------------------------------------------------------
def data_dictionary() -> pl.DataFrame:
    schema = pl.read_parquet_schema(DATA_DIR / TRAINING_FILES[0])
    tr = _scan_train().filter(pl.col(PHASE_COL) == DEP)
    rk = _scan_rank().filter(pl.col(PHASE_COL) == DEP)
    cols = list(schema)
    stats_tr = tr.select([pl.col(c).null_count().alias(f"null_{c}") for c in cols]
                         + [pl.col(c).n_unique().alias(f"nuniq_{c}") for c in cols] + [pl.len().alias("n")]).collect()
    stats_rk = rk.select([pl.col(c).null_count().alias(f"null_{c}") for c in cols]
                         + [pl.col(c).n_unique().alias(f"nuniq_{c}") for c in cols] + [pl.len().alias("n")]).collect()
    n_tr, n_rk = stats_tr["n"][0], stats_rk["n"][0]
    rows = []
    for c in cols:
        kind, desc = DICTIONARY[c]
        rows.append({
            "variable": c, "dtype": str(schema[c]), "tipo": kind,
            "null_pct_train_dep": round(100 * stats_tr[f"null_{c}"][0] / n_tr, 3),
            "null_pct_rank_dep": round(100 * stats_rk[f"null_{c}"][0] / n_rk, 3),
            "cardinalidad_train_dep": int(stats_tr[f"nuniq_{c}"][0]),
            "cardinalidad_rank_dep": int(stats_rk[f"nuniq_{c}"][0]),
            "oculta_en_ranking_dep": c in HIDDEN_IN_RANKING_DEP,
            "descripcion": desc,
        })
    return _save(pl.DataFrame(rows), "data_dictionary")


def top_values(k: int = 5) -> pl.DataFrame:
    """Valores más frecuentes de cada categórica (training-DEP)."""
    tr = _scan_train(list(dict.fromkeys(CAT_COLS + [PHASE_COL]))).filter(pl.col(PHASE_COL) == DEP)
    out = []
    for c in CAT_COLS:
        vc = tr.group_by(c).len().sort("len", descending=True).head(k).collect()
        for r in vc.iter_rows():
            out.append({"variable": c, "valor": (r[0] or "<null>")[:16], "n": r[1]})
    return _save(pl.DataFrame(out), "top_values_train_dep")


# ---------------------------------------------------------------------------
# 2. Distribución del target (DEP)
# ---------------------------------------------------------------------------
def target_distribution(tr: pl.DataFrame) -> pl.DataFrame:
    y = tr[TARGET]
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    summary = pl.DataFrame({
        "stat": ["n", "mean", "std", "min", "max"] + [f"q{q}" for q in qs],
        "value": [float(v) for v in [y.len(), y.mean(), y.std(), y.min(), y.max()] + [y.quantile(q) for q in qs]],
    })
    _save(summary, "target_summary_dep")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    clip = y.clip(0, 3600).to_numpy()
    ax[0].hist(clip, bins=120, color=C_TRAIN)
    ax[0].set(title="Taxi-out DEP (recortado a 0–3600 s solo para el gráfico)", xlabel="segundos", ylabel="movimientos")
    ax[0].axvline(y.median(), color=C_RANK, lw=1.5, label=f"mediana = {y.median():.0f} s")
    ax[0].legend()
    pos = y.filter(y > 0).to_numpy()
    ax[1].hist(np.log10(pos), bins=120, color=C_TRAIN)
    ax[1].set(title="log10(taxi-out) — solo valores > 0", xlabel="log10(segundos)")
    _fig("target_hist_dep")
    return summary


def outliers(tr: pl.DataFrame) -> pl.DataFrame:
    """Cuantifica negativos, ceros y extremos con varios umbrales. No elimina nada."""
    y = pl.col(TARGET)
    q1, q3 = tr[TARGET].quantile(0.25), tr[TARGET].quantile(0.75)
    iqr_hi = q3 + 3 * (q3 - q1)
    thresholds = {
        "negativos (<0)": y < 0, "ceros (==0)": y == 0, "muy cortos (0<y<60 s)": (y > 0) & (y < 60),
        "> 30 min": y > 1800, "> 60 min": y > 3600, "> 2 h": y > 7200, "> 3 h": y > 10800, "> 6 h": y > 21600,
        f"> Q3 + 3·IQR ({iqr_hi:.0f} s)": y > iqr_hi,
    }
    n = tr.height
    rows = [{"criterio": k, "n": tr.filter(v).height, "pct": round(100 * tr.filter(v).height / n, 4)} for k, v in thresholds.items()]
    out = _save(pl.DataFrame(rows), "outliers_summary_dep")

    # ¿Dónde se concentran? aeropuertos con más extremos (>2h) y con más negativos
    ext = tr.filter(y > 7200).group_by("ADEP_mvt").len().sort("len", descending=True).head(15).rename({"len": "n_gt_2h"})
    neg = tr.filter(y < 0).group_by("ADEP_mvt").len().sort("len", descending=True).head(15).rename({"len": "n_neg"})
    _save(ext, "outliers_gt2h_by_airport"); _save(neg, "outliers_negative_by_airport")

    # Serie temporal diaria de extremos: ¿son días concretos (tormentas, huelgas) o ruido?
    daily = (tr.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
               .group_by("day").agg(pl.len().alias("n"), (y > 3600).sum().alias("n_gt_1h"), y.median().alias("median"))
               .sort("day"))
    _save(daily, "target_daily_dep")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax.plot(daily["day"], daily["median"], color=C_TRAIN, lw=1.5)
    ax.set(title="Training DEP — mediana diaria del taxi-out", ylabel="segundos")
    ax2.plot(daily["day"], 100 * daily["n_gt_1h"] / daily["n"], color=C_RANK, lw=1)
    ax2.set(title="Proporción diaria de movimientos con taxi-out > 1 h", ylabel="%")
    _fig("target_daily_dep")
    return out


# ---------------------------------------------------------------------------
# 3. Cortes por variable
# ---------------------------------------------------------------------------
def _agg(tr: pl.DataFrame, by: list[str], name: str, top: int | None = None, min_n: int = 1) -> pl.DataFrame:
    y = pl.col(TARGET)
    g = (tr.group_by(by).agg(pl.len().alias("n"), y.mean().round(1).alias("mean"), y.median().alias("median"),
                             y.quantile(0.9).alias("p90"), y.std().round(1).alias("std"))
           .filter(pl.col("n") >= min_n).sort("n", descending=True))
    if top:
        g = g.head(top)
    return _save(g, name)


def by_airport(tr: pl.DataFrame, top: int = 30) -> pl.DataFrame:
    g = _agg(tr, ["ADEP_mvt"], "taxi_by_airport", top=None)
    t = g.head(top)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(t.height)
    ax.bar(x, t["median"], color=C_TRAIN, width=0.6, label="mediana")
    ax.scatter(x, t["p90"], color=C_RANK, s=14, zorder=3, label="p90")
    ax.set_xticks(x); ax.set_xticklabels(t["ADEP_mvt"], rotation=60, ha="right")
    ax.set(title=f"Taxi-out por aeropuerto de salida — {top} aeropuertos con más salidas", ylabel="segundos")
    ax.legend()
    _fig("taxi_by_airport_top")
    return g


def by_calendar(tr: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out = {k: _agg(tr, [k], f"taxi_by_{k}").sort(k) for k in ["month", "dow", "hour"]}
    out["is_weekend"] = _agg(tr, ["is_weekend"], "taxi_by_is_weekend").sort("is_weekend")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, k, lab in zip(axes, ["month", "dow", "hour"], ["mes", "día de semana (1=lun)", "hora UTC"]):
        d = out[k]
        ax.plot(d[k], d["median"], marker="o", ms=4, color=C_TRAIN, label="mediana")
        ax.plot(d[k], d["p90"], marker="o", ms=4, color=C_RANK, label="p90")
        ax.set(title=f"Taxi-out por {lab}", xlabel=lab, ylabel="segundos")
    axes[0].legend()
    _fig("taxi_by_calendar")
    # Volumen por hora: proxy de congestión
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(out["hour"]["hour"], out["hour"]["n"], color=C_TRAIN)
    ax.set(title="Nº de salidas por hora UTC (training)", xlabel="hora UTC", ylabel="movimientos")
    _fig("departures_by_hour")
    return out


def by_operational(tr: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out = {
        "runway": _agg(tr, ["ADEP_mvt", "RUNWAY_mvt"], "taxi_by_airport_runway", top=40, min_n=200),
        "stand": _agg(tr, ["ADEP_mvt", "STAND_mvt"], "taxi_by_airport_stand", top=40, min_n=200),
        "aircraft": _agg(tr, ["AIRCRAFT_TYPE_mvt"], "taxi_by_aircraft_type", top=30),
        "operator": _agg(tr, ["AIRCRAFT_OPERATOR_flt"], "taxi_by_operator", top=30),
        "segment": _agg(tr, ["MARKET_SEGMENT_flt"], "taxi_by_market_segment"),
        "wtc": _agg(tr, ["WK_TBL_CAT_flt"], "taxi_by_wake_category"),
    }
    # Varianza explicada (R² de "predecir con la mediana del grupo") para comparar poder de cada corte
    rows = []
    y = pl.col(TARGET)
    var_total = tr[TARGET].var()
    for label, keys in {"ADEP": ["ADEP_mvt"], "ADEP+RUNWAY": ["ADEP_mvt", "RUNWAY_mvt"], "ADEP+STAND": ["ADEP_mvt", "STAND_mvt"],
                        "ADEP+HOUR": ["ADEP_mvt", "hour"], "ADEP+AIRCRAFT": ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"],
                        "ADEP+RUNWAY+STAND": ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"],
                        "ADEP+RUNWAY+HOUR+AIRCRAFT": ["ADEP_mvt", "RUNWAY_mvt", "hour", "AIRCRAFT_TYPE_mvt"],
                        "OPERATOR": ["AIRCRAFT_OPERATOR_flt"], "SEGMENT": ["MARKET_SEGMENT_flt"]}.items():
        # varianza intra-grupo ponderada (in-sample: cota optimista del poder explicativo)
        g = tr.group_by(keys).agg(pl.len().alias("n"), y.var().fill_null(0).alias("v"), y.median().alias("med"))
        within = (g["n"] * g["v"]).sum() / g["n"].sum()
        # MAE in-sample usando la mediana del grupo
        mae = tr.join(g.select(keys + ["med"]), on=keys, how="left").select((y - pl.col("med")).abs().mean()).item()
        rows.append({"corte": label, "n_grupos": g.height, "R2_insample_media_grupo": round(1 - within / var_total, 4),
                     "MAE_insample_mediana_grupo": round(mae, 1)})
    out["explained"] = _save(pl.DataFrame(rows), "group_explained_variance")

    seg = out["segment"].sort("median", descending=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].barh(seg["MARKET_SEGMENT_flt"].fill_null("<null>"), seg["median"], color=C_TRAIN)
    ax[0].set(title="Mediana taxi-out por segmento de mercado", xlabel="segundos")
    ac = out["aircraft"].head(15)
    ax[1].barh(ac["AIRCRAFT_TYPE_mvt"].fill_null("<null>"), ac["median"], color=C_TRAIN)
    ax[1].invert_yaxis(); ax[1].set(title="Mediana taxi-out — 15 tipos de aeronave más frecuentes", xlabel="segundos")
    _fig("taxi_by_segment_aircraft")
    return out


# ---------------------------------------------------------------------------
# 4. Training vs ranking
# ---------------------------------------------------------------------------
def train_vs_ranking(tr: pl.DataFrame, rk: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    # Volumen mensual
    vt = tr.group_by("month").len().sort("month").with_columns(pl.lit("training 2025").alias("set"))
    vr = rk.group_by("month").len().sort("month").with_columns(pl.lit("ranking 2026").alias("set"))
    out["monthly"] = _save(pl.concat([vt, vr]), "volume_by_month_train_vs_rank")

    # Cobertura de categorías: ¿qué % de filas de ranking tiene una categoría vista en training?
    rows = []
    for c in ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]:
        keys = ["ADEP_mvt", c] if c in ("RUNWAY_mvt", "STAND_mvt") else [c]
        seen = tr.select(keys).unique()
        cov = rk.join(seen, on=keys, how="left", suffix="_seen").select(pl.len())  # count rows; use anti join for unseen
        unseen = rk.join(seen, on=keys, how="anti").height
        rows.append({"variable": "+".join(keys), "cat_train": seen.height, "cat_rank": rk.select(keys).unique().height,
                     "pct_filas_rank_no_vistas": round(100 * unseen / rk.height, 3)})
    out["coverage"] = _save(pl.DataFrame(rows), "category_coverage_rank_vs_train")

    # Cuota por aeropuerto (top 20)
    st = tr.group_by("ADEP_mvt").len().with_columns((pl.col("len") / tr.height * 100).alias("pct_train"))
    sr = rk.group_by("ADEP_mvt").len().with_columns((pl.col("len") / rk.height * 100).alias("pct_rank"))
    share = st.join(sr, on="ADEP_mvt", how="full", coalesce=True).fill_null(0).sort("pct_train", descending=True)
    out["airport_share"] = _save(share.select("ADEP_mvt", "pct_train", "pct_rank").head(40), "airport_share_train_vs_rank")
    t20 = out["airport_share"].head(20)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(t20.height); w = 0.4
    ax[0].bar(x - w / 2, t20["pct_train"], w, color=C_TRAIN, label="training 2025")
    ax[0].bar(x + w / 2, t20["pct_rank"], w, color=C_RANK, label="ranking 2026")
    ax[0].set_xticks(x); ax[0].set_xticklabels(t20["ADEP_mvt"], rotation=60, ha="right")
    ax[0].set(title="Cuota de salidas por aeropuerto (top 20)", ylabel="%"); ax[0].legend()
    ht = tr.group_by("hour").len().sort("hour"); hr = rk.group_by("hour").len().sort("hour")
    ax[1].plot(ht["hour"], ht["len"] / tr.height * 100, color=C_TRAIN, marker="o", ms=3, label="training")
    ax[1].plot(hr["hour"], hr["len"] / rk.height * 100, color=C_RANK, marker="o", ms=3, label="ranking")
    ax[1].set(title="Distribución horaria de salidas", xlabel="hora UTC", ylabel="%"); ax[1].legend()
    _fig("train_vs_rank_airport_hour")

    # Drift del taxi-IN (ARR) — el único target observable en ranking: 2025 vs 2026 por aeropuerto
    cols = [PHASE_COL, TARGET, "ADES_mvt"]
    a_tr = (_scan_train(cols).filter(pl.col(PHASE_COL) == "ARR").group_by("ADES_mvt")
            .agg(pl.len().alias("n_train"), pl.col(TARGET).median().alias("taxi_in_median_2025")).collect())
    a_rk = (_scan_rank(cols).filter(pl.col(PHASE_COL) == "ARR").group_by("ADES_mvt")
            .agg(pl.len().alias("n_rank"), pl.col(TARGET).median().alias("taxi_in_median_2026")).collect())
    drift = (a_tr.join(a_rk, on="ADES_mvt", how="inner").filter(pl.col("n_train") > 2000)
             .with_columns((pl.col("taxi_in_median_2026") - pl.col("taxi_in_median_2025")).alias("delta_s"))
             .sort("n_train", descending=True))
    out["taxi_in_drift"] = _save(drift, "taxi_in_drift_by_airport")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(drift["taxi_in_median_2025"], drift["taxi_in_median_2026"], s=drift["n_train"] / 800 + 6, color=C_TRAIN, alpha=0.7)
    lim = [0, max(drift["taxi_in_median_2025"].max(), drift["taxi_in_median_2026"].max()) * 1.05]
    ax.plot(lim, lim, color=C_GRAY, lw=1, ls="--")
    ax.set(title="Taxi-IN mediano por aeropuerto: 2025 (train) vs 2026 (ranking)", xlabel="2025 (s)", ylabel="2026 (s)")
    _fig("taxi_in_drift")

    # Nulos de las columnas _flt (enlace con plan de vuelo) en train vs rank
    flt_null = pl.DataFrame({
        "set": ["training DEP", "ranking DEP"],
        "pct_sin_plan_de_vuelo": [round(100 * tr["AOBT_3_flt"].null_count() / tr.height, 3), round(100 * rk["AOBT_3_flt"].null_count() / rk.height, 3)],
        "pct_aobt3_null": [round(100 * tr["AOBT_3_flt"].null_count() / tr.height, 3), round(100 * rk["AOBT_3_flt"].null_count() / rk.height, 3)],
    })
    out["flt_nulls"] = _save(flt_null, "flt_link_nulls_train_vs_rank")
    return out


# ---------------------------------------------------------------------------
# 5. Redundancia y leakage
# ---------------------------------------------------------------------------
def leakage_and_redundancy(tr: pl.DataFrame, rk: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    y = pl.col(TARGET)
    sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()  # noqa: E731

    # Diferencias temporales disponibles en ranking-DEP (todas las marcas salvo BLOCK_TIME)
    deltas = {
        "mvt_minus_aobt3": sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        "mvt_minus_eobt1": sec("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
        "mvt_minus_lobt": sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        "mvt_minus_iobt": sec("MVT_TIME_UTC_mvt", "IOBT_flt"),
        "mvt_minus_sched": sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "aobt3_minus_sched": sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
        "block_minus_aobt3 (LEAK: usa BLOCK)": sec("BLOCK_TIME_UTC_mvt", "AOBT_3_flt"),
    }
    d = tr.with_columns([v.alias(k) for k, v in deltas.items()])
    rows = []
    for k in deltas:
        col = d[k]
        valid = d.filter(pl.col(k).is_not_null())
        err = valid.select((pl.col(k) - y))  # error si se usara el delta como predicción directa del target
        rows.append({
            "delta": k, "pct_disponible": round(100 * (1 - col.null_count() / d.height), 2),
            "corr_con_target": round(valid.select(pl.corr(k, TARGET)).item(), 4),
            "MAE_como_predictor": round(err.select(pl.col(k).abs().mean()).item(), 1),
            "RMSE_como_predictor": round(float(np.sqrt(err.select((pl.col(k) ** 2).mean()).item())), 1),
            "pct_|error|<=60s": round(100 * err.filter(pl.col(k).abs() <= 60).height / valid.height, 2),
            "median_delta": valid[k].median(),
        })
    out["deltas"] = _save(pl.DataFrame(rows), "leakage_time_deltas")

    # Histograma del error de la baseline operacional MVT − AOBT_3
    e = d.filter(pl.col("mvt_minus_aobt3").is_not_null()).select((pl.col("mvt_minus_aobt3") - y).alias("e"))["e"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    # Bins de 5 s: las marcas de varios aeropuertos tienen resolución de 5 s y AOBT_3 de 60 s,
    # por lo que bins más anchos producen un peine artificial (aliasing).
    inside = e[(e >= -900) & (e <= 900)]
    ax[0].hist(inside, bins=np.arange(-900, 905, 5), color=C_TRAIN)
    ax[0].set(title=f"(MVT_TIME − AOBT_3) − taxi-out real  [|e| ≤ 900 s: {100*inside.size/e.size:.1f} % de filas]", xlabel="segundos")
    ax[0].axvline(0, color=C_RANK, lw=1)
    zoom = e[(e >= -60) & (e <= 60)]
    ax[1].hist(zoom, bins=np.arange(-60.5, 61.5, 1), color=C_TRAIN)
    ax[1].set(title=f"Zoom |e| ≤ 60 s ({100*zoom.size/e.size:.1f} % de filas), bins de 1 s", xlabel="segundos")
    _fig("baseline_aobt3_error")

    # Resolución temporal de las marcas por aeropuerto (fracción con segundos == 0:
    # 1/60 ≈ 0.017 → resolución 1 s; 1/12 ≈ 0.083 → resolución 5 s; ≈1.0 → resolución 1 min)
    res = tr.group_by("ADEP_mvt").agg(
        (pl.col("BLOCK_TIME_UTC_mvt").dt.second() == 0).mean().round(3).alias("BLOCK_TIME_sec0"),
        (pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).mean().round(3).alias("MVT_TIME_sec0"),
        (pl.col("AOBT_3_flt").dt.second() == 0).mean().round(3).alias("AOBT_3_sec0"),
        (pl.col("EOBT_1_flt").dt.second() == 0).mean().round(3).alias("EOBT_1_sec0"),
    ).sort("ADEP_mvt")
    out["resolution"] = _save(res, "timestamp_resolution_by_airport")

    # Redundancia entre pares de columnas (train DEP): % de igualdad
    pairs = [("ADEP_mvt", "ADEP_flt"), ("ADES_mvt", "ADES_flt"), ("ADES_flt", "ADES_FILED_flt"),
             ("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), ("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"),
             ("SCHED_TIME_UTC_mvt", "LOBT_flt"), ("SCHED_TIME_UTC_mvt", "EOBT_1_flt"), ("LOBT_flt", "EOBT_1_flt"),
             ("IOBT_flt", "EOBT_1_flt"), ("ARVT_1_flt", "ARVT_3_flt")]
    full = _scan_train([PHASE_COL] + sorted({c for p in pairs for c in p})).filter(pl.col(PHASE_COL) == DEP)
    exprs = []
    for a, b in pairs:
        both = pl.col(a).is_not_null() & pl.col(b).is_not_null()
        exprs.append(((pl.col(a) == pl.col(b)) & both).sum().alias(f"eq_{a}|{b}"))
        exprs.append(both.sum().alias(f"n_{a}|{b}"))
    r = full.select(exprs).collect()
    out["redundancy"] = _save(pl.DataFrame([{"col_a": a, "col_b": b, "pct_iguales_(ambas_no_nulas)": round(100 * r[f"eq_{a}|{b}"][0] / max(r[f"n_{a}|{b}"][0], 1), 2)}
                                            for a, b in pairs]), "redundancy_pairs")

    # Clasificación de riesgo de leakage por variable
    risk = [
        ("BLOCK_TIME_UTC_mvt", "PROHIBIDA", "Nula en ranking-DEP. Target = MVT_TIME − BLOCK_TIME exactamente."),
        ("TAXITIME_SEC_mvt", "TARGET", "Nula en ranking-DEP."),
        ("AOBT_3_flt", "ALTO (permitida)", "AOBT real a resolución de minuto. MVT_TIME − AOBT_3 ≈ target. Disponible en ranking: es la baseline operacional, no leakage según las reglas, pero domina cualquier modelo."),
        ("MVT_TIME_UTC_mvt", "MEDIO (permitida)", "Hora de despegue real: es 'futuro' respecto al off-block pero se da como input. Necesaria para todas las diferencias temporales."),
        ("ARVT_1_flt / ARVT_3_flt", "BAJO", "Hora de llegada al destino: posterior al despegue; sólo aporta duración de vuelo/ruta. Sin relación directa con el taxi-out."),
        ("EOBT_1_flt / LOBT_flt / IOBT_flt / SCHED_TIME", "BAJO", "Horas planificadas. MVT_TIME − EOBT ≈ taxi-out + retraso de salida: útil como fallback cuando falta AOBT_3."),
        ("MVT_ID_mvt / FLIGHT_ID_mvt", "IDENTIFICADOR", "Correlacionan con el tiempo (ids crecientes). No usar como predictor."),
        ("Filas ARR del ranking", "CONTEXTO", "Taxi-in real 2026 disponible: puede usarse para features de congestión del aeropuerto en la misma hora (no es leakage: es input del ranking)."),
    ]
    out["risk"] = _save(pl.DataFrame({"variable": [r[0] for r in risk], "riesgo": [r[1] for r in risk], "motivo": [r[2] for r in risk]}), "leakage_risk_matrix")

    # Correlación numérica simple entre deltas disponibles y el target (train-DEP)
    return out


def congestion_signal(tr: pl.DataFrame) -> pl.DataFrame:
    """Nº de salidas del mismo aeropuerto en la misma hora vs taxi-out (proxy de cola en pista)."""
    g = tr.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("h")) \
          .group_by(["ADEP_mvt", "h"]).agg(pl.len().alias("dep_per_hour"), pl.col(TARGET).median().alias("median_taxi"))
    top = tr.group_by("ADEP_mvt").len().sort("len", descending=True).head(8)["ADEP_mvt"].to_list()
    rows = []
    for a in top:
        s = g.filter(pl.col("ADEP_mvt") == a)
        rows.append({"ADEP_mvt": a, "corr_spearman(dep_per_hour, median_taxi)": round(s.select(pl.corr("dep_per_hour", "median_taxi", method="spearman")).item(), 3)})
    out = _save(pl.DataFrame(rows), "congestion_hourly_corr")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=False)
    for ax, a in zip(axes.ravel(), top):
        s = g.filter(pl.col("ADEP_mvt") == a)
        b = s.group_by("dep_per_hour").agg(pl.col("median_taxi").median().alias("m"), pl.len().alias("n")).filter(pl.col("n") >= 20).sort("dep_per_hour")
        ax.plot(b["dep_per_hour"], b["m"], color=C_TRAIN, marker="o", ms=3)
        ax.set_title(a, fontsize=10); ax.set_xlabel("salidas/hora"); ax.set_ylabel("mediana taxi-out (s)")
    fig.suptitle("Congestión: salidas por hora vs mediana del taxi-out (8 aeropuertos mayores)")
    _fig("congestion_vs_taxi")
    return out
