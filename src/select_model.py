"""
Selección del modelo final a partir de las predicciones de validación ya guardadas
(`models/backtest_predictions/*.npy`), sin reentrenar ni optimizar pesos.

Candidatos (estrategia fija `--strategy`, por defecto drop_nonpositive):
  A) catboost_direct
  B) catboost_residual
  C) ensamble 50/50 = 0.5·A + 0.5·B   (peso fijo a priori: NO se optimiza sobre los mismos folds)

Criterio: menor RMSE medio de los 3 folds (la métrica oficial del concurso es RMSE).
También se evalúa si recortar predicciones negativas a 0 antes de la submission mejora o
empeora el RMSE en los backtests.

Salidas: outputs/tables/ensemble_backtest.csv, outputs/tables/negative_clip_backtest.csv,
         models/final/selection.json
Uso:     python src/select_model.py [--strategy drop_nonpositive]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import FOLDS, MODELS_DIR, OUTLIER_STRATEGIES, TABLES_DIR  # noqa: E402
from modeling_utils import mae, rmse  # noqa: E402

PRED_DIR = MODELS_DIR / "backtest_predictions"
ENSEMBLE_WEIGHT = 0.5  # fijado a priori


def load_preds(strategy: str) -> dict:
    out = {}
    for fold in FOLDS:
        y = np.load(PRED_DIR / f"y__{fold}.npy").astype(np.float64)
        a = np.load(PRED_DIR / f"{fold}__{strategy}__catboost_direct.npy").astype(np.float64)
        b = np.load(PRED_DIR / f"{fold}__{strategy}__catboost_residual.npy").astype(np.float64)
        assert y.shape == a.shape == b.shape
        out[fold] = (y, a, b)
    return out


def evaluate(strategy: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, clip_rows = [], []
    for fold, (y, a, b) in load_preds(strategy).items():
        cands = {"catboost_direct": a, "catboost_residual": b, "ensemble_50_50": ENSEMBLE_WEIGHT * a + (1 - ENSEMBLE_WEIGHT) * b}
        for name, p in cands.items():
            rows.append({"fold": fold, "strategy": strategy, "candidate": name, "rmse": rmse(y, p), "mae": mae(y, p),
                         "n_pred_negative": int((p < 0).sum())})
            pc = np.maximum(p, 0.0)
            clip_rows.append({"fold": fold, "candidate": name, "rmse_raw": rmse(y, p), "rmse_clipped": rmse(y, pc),
                              "delta_rmse_clip": rmse(y, pc) - rmse(y, p), "n_clipped": int((p < 0).sum()),
                              "n_y_negative": int((y < 0).sum())})
    return pl.DataFrame(rows), pl.DataFrame(clip_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="drop_nonpositive", choices=OUTLIER_STRATEGIES)
    args = ap.parse_args()
    res, clip = evaluate(args.strategy)
    summary = res.group_by("candidate").agg(pl.col("rmse").mean().alias("rmse_mean"), pl.col("mae").mean().alias("mae_mean"),
                                            pl.col("rmse").std().alias("rmse_std")).sort("rmse_mean")
    full = res.join(summary, on="candidate", how="left").sort("candidate", "fold")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    full.write_csv(TABLES_DIR / "ensemble_backtest.csv")
    clip.write_csv(TABLES_DIR / "negative_clip_backtest.csv")
    # Solo se compara directo vs ensamble (el residual solo participa dentro del ensamble)
    cmp = summary.filter(pl.col("candidate").is_in(["catboost_direct", "ensemble_50_50"])).sort("rmse_mean")
    chosen = cmp["candidate"][0]
    clip_summary = clip.group_by("candidate").agg(pl.col("delta_rmse_clip").mean(), pl.col("n_clipped").sum()).sort("candidate")
    clip_row = clip_summary.filter(pl.col("candidate") == chosen)
    clip_negatives = bool(clip_row["delta_rmse_clip"][0] <= 0)
    per_fold = {c: {r["fold"]: round(r["rmse"], 3) for r in res.filter(pl.col("candidate") == c).iter_rows(named=True)} for c in summary["candidate"]}
    sel = {
        "selected": chosen, "strategy": args.strategy, "ensemble_weight_direct": ENSEMBLE_WEIGHT if chosen == "ensemble_50_50" else None,
        "criterion": "menor RMSE medio de los 3 backtests entre catboost_direct y ensemble_50_50 (peso 0.5 fijado a priori, no optimizado)",
        "rmse_mean": {r["candidate"]: round(r["rmse_mean"], 3) for r in summary.iter_rows(named=True)},
        "mae_mean": {r["candidate"]: round(r["mae_mean"], 3) for r in summary.iter_rows(named=True)},
        "rmse_by_fold": per_fold,
        "clip_negative_predictions": clip_negatives,
        "clip_evidence": {r["candidate"]: {"mean_delta_rmse": round(r["delta_rmse_clip"], 4), "n_clipped": int(r["n_clipped"])} for r in clip_summary.iter_rows(named=True)},
        "bundles": {"catboost_direct": "models/final/catboost_direct", "catboost_residual": "models/final/catboost_residual"},
        "decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "final").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "final" / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary); print(cmp); print(clip_summary)
    print(f"\nSeleccionado: {chosen} | recortar negativos: {clip_negatives} → models/final/selection.json")


if __name__ == "__main__":
    main()
