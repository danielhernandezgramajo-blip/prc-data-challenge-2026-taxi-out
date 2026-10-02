"""
V4 — Definición de las anclas de BLOCK (fuentes visibles) y atribución por fila.

Un "ancla" es un timestamp visible en ranking (más un desplazamiento de ±1 día) del que BLOCK_TIME_UTC_mvt podría haberse
copiado. Este módulo NO lee el target: `anchor_frame` solo usa columnas visibles; `attribute` recibe BLOCK explícitamente y
solo se usa con filas de entrenamiento (o para informes sobre filas de validación).
"""
from __future__ import annotations

import numpy as np
import polars as pl

DAY = 86_400.0
SOURCES = ["SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]   # orden = precedencia en empates
SHORT = {"SCHED_TIME_UTC_mvt": "SCHED", "IOBT_flt": "IOBT", "LOBT_flt": "LOBT", "EOBT_1_flt": "EOBT1", "AOBT_3_flt": "AOBT3"}
SHIFTS = {"": 0.0, "-1d": -DAY, "+1d": DAY}
ANCHORS = [SHORT[s] + k for s in SOURCES for k in SHIFTS]          # 15 anclas, en orden de precedencia
ANCHOR_SRC = {SHORT[s] + k: (s, v) for s in SOURCES for k, v in SHIFTS.items()}
TOLS = (10, 30, 60, 120, 300)


def anchor_frame(df: pl.DataFrame) -> pl.DataFrame:
    """[MVT_ID] + epoch (s, float) de cada ancla (null si la fuente falta) + MVT. Sin target."""
    exprs = [pl.col("MVT_ID_mvt"), pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_mvt")]
    for a in ANCHORS:
        src, sh = ANCHOR_SRC[a]
        exprs.append((pl.col(src).dt.epoch("s").cast(pl.Float64) + sh).alias(f"ts_{a}"))
    return df.select(exprs)


def anchor_matrix(af: pl.DataFrame) -> np.ndarray:
    return np.column_stack([af[f"ts_{a}"].to_numpy().astype(np.float64) for a in ANCHORS])


def attribute(ts: np.ndarray, block: np.ndarray, tol: float) -> dict:
    """Atribución por fila. ts: (n, A) epoch de anclas (nan si ausente); block: (n,) epoch de BLOCK.
    Devuelve el ancla más cercana (índice, -1 si ninguna disponible), su distancia, la segunda (ancla distinta), el número
    de anclas dentro de `tol`, si hay ambigüedad (≥ 2 anclas dentro de tol con timestamps que difieren > 10 s) y la clase
    atribuida (índice del ancla más cercana si |d| ≤ tol, si no -1). Empates: el orden de ANCHORS (precedencia)."""
    d = block[:, None] - ts
    ad = np.where(np.isnan(d), np.inf, np.abs(d))
    n_avail = np.isfinite(ad).sum(axis=1)
    order = np.argsort(ad, axis=1, kind="stable")          # estable → en empates gana la precedencia de ANCHORS
    i1 = order[:, 0]; i2 = order[:, 1]
    r = np.arange(len(block))
    d1 = ad[r, i1]; d2 = ad[r, i2]
    within = ad <= tol
    n_within = within.sum(axis=1)
    ts_w = np.where(within, ts, np.nan)
    spread = np.nanmax(ts_w, axis=1, initial=-np.inf) - np.nanmin(ts_w, axis=1, initial=np.inf)
    ambiguous = (n_within >= 2) & (spread > 10)
    nearest = np.where(n_avail > 0, i1, -1)
    cls = np.where(d1 <= tol, i1, -1)
    return {"nearest": nearest, "dist1": np.where(np.isfinite(d1), d1, np.nan), "second": np.where(np.isfinite(d2), i2, -1),
            "dist2": np.where(np.isfinite(d2), d2, np.nan), "n_within": n_within, "ambiguous": ambiguous, "cls": cls,
            "signed_d1": np.where(n_avail > 0, d[r, i1], np.nan)}


def align_by_id(ids: np.ndarray, af: pl.DataFrame, id_col: str = "MVT_ID_mvt") -> pl.DataFrame:
    """Filas de `af` en el orden EXACTO de `ids`. MVT_ID se usa solo como clave de alineación (nunca como valor).
    Exige ids enteros y únicos en ambos lados y que todos los ids existan en `af`; si no, ValueError."""
    ids = np.asarray(ids)
    if ids.dtype.kind == "f" and (not np.all(np.isfinite(ids)) or not np.all(ids == np.round(ids))):
        raise ValueError("ids no enteros")
    ids_i = ids.astype(np.int64)
    if np.unique(ids_i).size != ids_i.size:
        raise ValueError("ids duplicados en la consulta")
    right = af.with_columns(pl.col(id_col).cast(pl.Int64), pl.lit(True).alias("_hit"))
    if right[id_col].is_duplicated().any():
        raise ValueError("ids duplicados en af")
    j = pl.DataFrame({id_col: ids_i, "_o": np.arange(ids_i.size)}).join(right, on=id_col, how="left").sort("_o")
    missing = j["_hit"].null_count()
    if missing:
        raise ValueError(f"{missing} ids sin fila en af")
    return j.drop("_o", "_hit")


def expected_block(p: np.ndarray, ts: np.ndarray, block_none: np.ndarray, resid: np.ndarray | None = None,
                   p_ds: np.ndarray | None = None) -> np.ndarray:
    """E[BLOCK|x] de la mezcla prefijada en protocol.md §4:
        (1 − p_ds)·[Σ_k p_k·(ts_k + r̂_k) + p_none·BLOCK_none] + p_ds·(BLOCK_none − 86 400)
    p: (n, K+1) probabilidades de las K anclas + 'none' (última columna); ts: (n, K) epoch de las anclas (nan si falta);
    block_none: (n,) BLOCK del experto 'none'; resid: (n, K) corrección por ancla (0 si None); p_ds: (n,) clase de fecha
    desplazada. Anclas ausentes → p_k = 0 y renormalización sobre las disponibles + none; si toda la masa estaba en anclas
    ausentes, pasa entera a 'none'."""
    p = np.array(p, dtype=np.float64, copy=True)
    n, k1 = p.shape
    K = k1 - 1
    if ts.shape != (n, K):
        raise ValueError("ts debe ser (n, K) con K = columnas de p − 1")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilidades negativas o no finitas")
    avail = np.isfinite(ts)
    p[:, :K] = np.where(avail, p[:, :K], 0.0)
    tot = p.sum(axis=1)
    p[tot <= 0, K] = 1.0
    p /= p.sum(axis=1, keepdims=True)
    r = np.zeros_like(ts, dtype=np.float64) if resid is None else np.asarray(resid, dtype=np.float64)
    e = (p[:, :K] * np.where(avail, ts + r, 0.0)).sum(axis=1) + p[:, K] * block_none
    if p_ds is not None:
        e = (1.0 - p_ds) * e + p_ds * (block_none - DAY)
    return e


def copy_mass(d: np.ndarray, n_total: int, inner: float = 10.0, sh_lo: float = 30.0, sh_hi: float = 120.0) -> float:
    """Exceso de filas con |d| ≤ inner sobre el fondo local estimado en los hombros sh_lo < |d| ≤ sh_hi (densidad por
    segundo entero), como fracción de n_total. Distingue copias reales de cercanías casuales."""
    d = d[np.isfinite(d)]
    a = np.abs(d)
    n_in = float((a <= inner).sum())
    dens = float(((a > sh_lo) & (a <= sh_hi)).sum()) / (2.0 * (sh_hi - sh_lo))
    return (n_in - (2 * inner + 1) * dens) / n_total
