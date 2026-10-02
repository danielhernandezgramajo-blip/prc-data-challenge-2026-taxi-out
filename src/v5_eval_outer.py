"""
V5 — Comprobación única en enero/julio 2025 (outer) de la mezcla de ramas normales CatBoost + LightGBM frente a C2 (V4b),
con la misma compuerta y celda de V4b (variante S). Regla fijada en outputs/experiments/v5/protocol.md §2.

    python src/v5_eval_outer.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from config import MODELS_DIR, OUTPUTS_DIR  # noqa: E402
from v4b_analysis import W, weighted_block_bootstrap  # noqa: E402
from v5_eval_inner import combine  # noqa: E402

V4B = MODELS_DIR / "experiments" / "v4b" / "preds"
V5 = MODELS_DIR / "experiments" / "v5" / "preds"
SEEDS = (42, 7, 2025)
MONTH = {"LOJO": "ene", "LOJuO": "jul"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="§2b: rama mezclada + compuerta mezclada frente a rama mezclada")
    args = ap.parse_args()
    base, cand = ("blend_S", "blend_gate_S") if args.gate else ("C2_S", "blend_S")
    errs, blocks, res = {}, {}, {"by_month": {}, "comparison": f"{cand} vs {base}"}
    for f in W:
        m = np.load(V4B / f"meta__{f}__outer.npz", allow_pickle=True)
        y = m["y"].astype(np.float64); s = m["s"]; cell = m["in_cell"]; rc = m["regime_class"]
        lgb = np.load(V5 / f"lgb_const__{f}__s42__outer.npy").astype(np.float64)
        pg = np.load(V5 / f"lgb_gate__{f}__s42__outer.npy").astype(np.float64) if args.gate else None
        c2, bl, bg = [], [], []
        for sd in SEEDS:
            b = np.load(V4B / f"c2_n__{f}__s{sd}__outer.npy").astype(np.float64)
            p1 = np.load(V4B / f"c1_clf__{f}__s{sd}__outer.npy").astype(np.float64)
            P0 = np.load(V4B / f"c0_clf__{f}__s{sd}__outer.npy").astype(np.float64)
            c2.append(combine(b, p1, P0, s, cell)); bl.append(combine(0.5 * b + 0.5 * lgb, p1, P0, s, cell))
            if args.gate:
                bg.append(combine(0.5 * b + 0.5 * lgb, 0.5 * p1 + 0.5 * pg, P0, s, cell))
        E = {"C2_S": (np.mean(c2, axis=0) - y) ** 2, "blend_S": (np.mean(bl, axis=0) - y) ** 2}
        if args.gate:
            E["blend_gate_S"] = (np.mean(bg, axis=0) - y) ** 2
        bulk = (rc == 0) & (y <= 3600)
        res["by_month"][MONTH[f]] = {"n": int(y.size), **{k: float(np.sqrt(v.mean())) for k, v in E.items()},
                                     "delta": float(np.sqrt(E[cand].mean()) - np.sqrt(E[base].mean())),
                                     **{f"bulk_{k}": float(np.sqrt(v[bulk].mean())) for k, v in E.items()},
                                     **{f"clase1_{k}": float(np.sqrt(v[rc == 1].mean())) for k, v in E.items()}}
        errs[f] = E
        blocks[f] = np.char.add(np.char.add(m["airport"].astype(str), "|"), m["day"].astype(str))
    rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in errs["LOJO"]}
    bs = weighted_block_bootstrap(errs, blocks, 1000, 42)
    d = bs[cand] - bs[base]
    res["RMSE_w"] = rw
    res["delta_RMSE_w"] = rw[cand] - rw[base]
    res["bootstrap_delta_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    both = all(v["delta"] < 0 for v in res["by_month"].values())
    res["rule"] = "ΔRMSE_w ≤ −1 s y mejora en ene y jul"
    res["pass"] = bool(res["delta_RMSE_w"] <= -1.0 and both)
    out = OUTPUTS_DIR / "experiments" / "v5" / ("outer_check_gate.json" if args.gate else "outer_check.json")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
