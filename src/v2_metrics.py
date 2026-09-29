"""V2 — métricas de regresión, cobertura y agregación pooled (sin promediar promedios)."""
from __future__ import annotations

import numpy as np
import polars as pl

from config import OUTLIER_HI_SEC

COVER_S = (300, 600, 900)
QUANTS = (0.5, 0.87, 0.9, 0.95, 0.99)


def block(y: np.ndarray, p: np.ndarray) -> dict:
    """Todas las métricas pedidas para un vector de reales y predicciones."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    e = p - y; ae = np.abs(e)
    out = {"n": int(y.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(ae.mean()), "medae": float(np.median(ae)),
           "bias": float(e.mean())}
    for s in COVER_S:
        out[f"cov_le{s}"] = float(100.0 * (ae <= s).mean())
    for q in QUANTS:
        out[f"ae_p{int(q*100)}"] = float(np.quantile(ae, q))
    out["ae_max"] = float(ae.max())
    out["n_pred_negative"] = int((p < 0).sum()); out["pct_pred_negative"] = float(100.0 * (p < 0).mean())
    normal = (y > 0) & (y <= OUTLIER_HI_SEC)
    out["n_normal"] = int(normal.sum()); out["n_extreme"] = int((y > OUTLIER_HI_SEC).sum())
    out["rmse_normal"] = float(np.sqrt(np.mean(e[normal] ** 2))) if normal.any() else np.nan
    out["mae_normal"] = float(ae[normal].mean()) if normal.any() else np.nan
    out["cov_le300_normal"] = float(100.0 * (ae[normal] <= 300).mean()) if normal.any() else np.nan
    ext = y > OUTLIER_HI_SEC
    out["rmse_extreme"] = float(np.sqrt(np.mean(e[ext] ** 2))) if ext.any() else np.nan
    out["mae_extreme"] = float(ae[ext].mean()) if ext.any() else np.nan
    # contribución de los extremos (>3 h) al MSE total
    out["mse_share_extreme_pct"] = float(100.0 * (e[ext] ** 2).sum() / (e ** 2).sum()) if ext.any() else 0.0
    return out


def pooled(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Métricas pooled: concatena todas las filas de todos los folds (RMSE = sqrt(mean de TODOS los errores²))."""
    y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
    return block(y, p)


def by_airport(airport: np.ndarray, y: np.ndarray, p: np.ndarray) -> pl.DataFrame:
    rows = []
    for a in sorted(set(airport.tolist())):
        m = airport == a
        rows.append({"ADEP_mvt": a, **block(y[m], p[m])})
    return pl.DataFrame(rows)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))
