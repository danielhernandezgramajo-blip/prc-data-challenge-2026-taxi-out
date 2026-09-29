"""
V2 · Paso 1 — Análisis de regímenes de la cola extrema (sin entrenar nada, sin tocar V1).

Preguntas:
 1. ¿Cuántos extremos (> 3 h) hay, dónde y con qué información de plan de vuelo?
 2. Cuando AOBT_3 existe, ¿el taxi real ≈ MVT − AOBT_3 también en los extremos? (residual r = y − (MVT − AOBT_3))
 3. Regímenes OBSERVABLES en ranking (has_aobt3, tamaño de MVT − AOBT_3, de MVT − SCHED, de MVT − LOBT):
    n, P(y > 3 h), media/mediana de y, cuánto pesa cada régimen en el MSE, y el valor constante óptimo (media).
 4. Con las predicciones de validación ya guardadas (backtests V1 jul/nov/dic, drop_nonpositive): error del
    modelo directo, del residual y del 50/50 por régimen → qué peso conviene en cada régimen.

Salidas: outputs/experiments/v2/tail_*.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from backtest import split_fold as split_fold_v1  # noqa: E402
from config import FOLDS, MODELS_DIR, OUTLIER_HI_SEC, OUTPUTS_DIR, TARGET  # noqa: E402
from prepare_training import load_features  # noqa: E402

OUT = OUTPUTS_DIR / "experiments" / "v2"
H = 3600.0


def regime_expr() -> pl.Expr:
    """Régimen observable en ranking (solo features disponibles)."""
    a = pl.col("mvt_minus_aobt3"); s = pl.col("mvt_minus_sched"); l = pl.col("mvt_minus_lobt")  # noqa: E741
    return (
        pl.when(a.is_null() & (s > 6 * H)).then(pl.lit("R6_noFPL_sched>6h"))
          .when(a.is_null()).then(pl.lit("R5_noFPL"))
          .when(a > 6 * H).then(pl.lit("R4_aobt3>6h"))
          .when(a > 2 * H).then(pl.lit("R3_aobt3_2-6h"))
          .when((a <= 2 * H) & (s > 6 * H)).then(pl.lit("R2_aobt3<2h_sched>6h"))
          .when((a <= 2 * H) & (l > 3 * H)).then(pl.lit("R2b_aobt3<2h_lobt>3h"))
          .otherwise(pl.lit("R1_normal"))
    ).alias("regime")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_features("train").with_columns(regime_expr())
    y = pl.col(TARGET)
    ext = y > OUTLIER_HI_SEC

    # 1. Inventario de extremos
    inv = (tr.filter(ext).group_by("ADEP_mvt").agg(pl.len().alias("n_ext"), pl.col("has_aobt3").mean().round(3).alias("pct_with_aobt3"),
                                                   y.median().alias("median_y"), y.max().alias("max_y"))
             .sort("n_ext", descending=True))
    inv.write_csv(OUT / "tail_by_airport.csv"); print(inv)

    # 2. Coherencia de AOBT_3 en los extremos
    d = tr.with_columns((y - pl.col("mvt_minus_aobt3")).alias("r"))
    coh = (d.with_columns(pl.when(ext).then(pl.lit(">3h")).when(y > H).then(pl.lit("1-3h")).otherwise(pl.lit("<=1h")).alias("y_bucket"))
             .filter(pl.col("has_aobt3") == 1)
             .group_by("y_bucket").agg(pl.len().alias("n"), (pl.col("r").abs() <= 600).mean().round(3).alias("pct_|r|<=10min"),
                                       (pl.col("r").abs() <= 1800).mean().round(3).alias("pct_|r|<=30min"),
                                       pl.col("r").median().alias("median_r"), pl.col("r").quantile(0.1).alias("r_p10"), pl.col("r").quantile(0.9).alias("r_p90"))
             .sort("y_bucket"))
    coh.write_csv(OUT / "tail_aobt3_coherence.csv"); print(coh)

    # 3. Regímenes: tamaño, prevalencia de extremos, media (= constante RMSE-óptima), peso en el MSE de la mediana global
    med = float(tr[TARGET].median())
    reg = (d.group_by("regime").agg(
        pl.len().alias("n"), (pl.len() / tr.height * 100).round(4).alias("pct_rows"),
        ext.mean().round(4).alias("p_extreme"), y.mean().round(0).alias("mean_y"), y.median().alias("median_y"), y.quantile(0.9).alias("y_p90"),
        pl.col("mvt_minus_aobt3").mean().round(0).alias("mean_mvt_minus_aobt3"),
        ((y - med) ** 2).sum().alias("_sse_med"),
        # error de dos predictores "ingenuos" dentro del régimen: la media del régimen y MVT − AOBT_3
        ((y - y.mean()) ** 2).mean().sqrt().round(0).alias("rmse_regime_mean"),
        pl.when(pl.col("has_aobt3") == 1).then((y - pl.col("mvt_minus_aobt3")) ** 2).mean().sqrt().round(0).alias("rmse_mvt_minus_aobt3"),
    ).with_columns((pl.col("_sse_med") / pl.col("_sse_med").sum() * 100).round(2).alias("pct_of_SSE_vs_global_median")).drop("_sse_med")
     .sort("regime"))
    reg.write_csv(OUT / "tail_regimes.csv"); print(reg)

    # 4. Error de directo / residual / 50-50 por régimen con las predicciones de validación V1 (jul/nov/dic)
    pred_dir = MODELS_DIR / "backtest_predictions"
    rows = []
    for fold in FOLDS:
        _, val = split_fold_v1(tr, fold)
        yv = val[TARGET].to_numpy().astype(np.float64)
        y_saved = np.load(pred_dir / f"y__{fold}.npy").astype(np.float64)
        assert np.allclose(yv, y_saved), "orden de validación no reproducido"
        pd_ = np.load(pred_dir / f"{fold}__drop_nonpositive__catboost_direct.npy").astype(np.float64)
        pr_ = np.load(pred_dir / f"{fold}__drop_nonpositive__catboost_residual.npy").astype(np.float64)
        regs = val["regime"].to_numpy()
        for r in sorted(set(regs.tolist())):
            m = regs == r
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                p = w * pd_[m] + (1 - w) * pr_[m]
                rows.append({"fold": fold, "regime": r, "w_direct": w, "n": int(m.sum()), "n_extreme": int((yv[m] > OUTLIER_HI_SEC).sum()),
                             "rmse": float(np.sqrt(np.mean((p - yv[m]) ** 2))), "sse": float(((p - yv[m]) ** 2).sum())})
    res = pl.DataFrame(rows)
    res.write_csv(OUT / "tail_model_errors_by_regime.csv")
    # SSE pooled por régimen y peso (3 folds) → peso óptimo de la cuadrícula por régimen y ahorro vs 0.5
    pooled = (res.group_by("regime", "w_direct").agg(pl.col("sse").sum().alias("sse"), pl.col("n").sum().alias("n"), pl.col("n_extreme").sum().alias("n_extreme"))
                .with_columns((pl.col("sse") / pl.col("n")).sqrt().round(1).alias("rmse_pooled")).sort("regime", "w_direct"))
    best = pooled.group_by("regime").agg(pl.col("w_direct").sort_by("sse").first().alias("best_w"), pl.col("sse").min().alias("sse_best"))
    half = pooled.filter(pl.col("w_direct") == 0.5).select("regime", pl.col("sse").alias("sse_half"), "n", "n_extreme")
    summary = (best.join(half, on="regime").with_columns(((pl.col("sse_half") - pl.col("sse_best")) / 1e6).round(1).alias("sse_saved_M"))
                   .sort("regime"))
    total_n = int(res.filter(pl.col("w_direct") == 0.5)["n"].sum()); total_sse_half = float(res.filter(pl.col("w_direct") == 0.5)["sse"].sum())
    summary = summary.with_columns((pl.col("sse_saved_M") * 1e6 / total_sse_half * 100).round(2).alias("pct_of_total_SSE_50_50"))
    summary.write_csv(OUT / "tail_regime_weights_pooled.csv"); print(pooled); print(summary)
    print(f"RMSE pooled 50/50 (3 folds): {np.sqrt(total_sse_half / total_n):.1f}  →  con el mejor peso por régimen (in-sample, cota optimista): "
          f"{np.sqrt((total_sse_half - summary['sse_saved_M'].sum() * 1e6) / total_n):.1f}")


if __name__ == "__main__":
    main()
