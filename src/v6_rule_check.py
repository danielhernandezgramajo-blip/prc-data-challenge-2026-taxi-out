"""
V6 §1 — Comprobación fuera de mes de la regla de hueco LOBT sobre la rama V5a (outputs/experiments/v6/protocol.md §1).
Para cada fold la recta se ajusta SOLO con sus 9 meses de entrenamiento; se evalúa en el mes outer (ene/jul) y en el inner
(feb/ago). Umbral principal 2 h; 1.5 h y 3 h como sensibilidad; MVT − AOBT_3 > 1 h como contraste.

    python src/v6_rule_check.py           # → outputs/experiments/v6/rule_check.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import MODELS_DIR, OUTPUTS_DIR, TARGET  # noqa: E402

TRM = {"LOJO": [3, 4, 5, 6, 7, 9, 10, 11, 12], "LOJuO": [1, 3, 4, 5, 6, 9, 10, 11, 12]}
VARIANTS = (("mvt_minus_lobt", 7200.0, "principal"), ("mvt_minus_lobt", 5400.0, "sensibilidad"),
            ("mvt_minus_lobt", 10800.0, "sensibilidad"), ("mvt_minus_aobt3", 3600.0, "contraste"))


def main() -> None:
    from v4b_analysis import W, weighted_block_bootstrap
    from v5_eval_inner import combine
    from v6_final import apply_lobt_rule, fit_lobt_rule
    P = MODELS_DIR / "experiments" / "v4b" / "preds"
    V5 = MODELS_DIR / "experiments" / "v5" / "preds"
    F = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet",
                        columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", TARGET, "mvt_minus_lobt", "mvt_minus_aobt3"]
                        ).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    base_pred, meta = {}, {}
    for f in W:
        for part in ("outer", "inner"):
            mt = np.load(P / f"meta__{f}__{part}.npz", allow_pickle=True)
            lgb = np.load(V5 / f"lgb_const__{f}__s42__{part}.npy").astype(np.float64)
            base_pred[(f, part)] = np.mean([combine(0.5 * np.load(P / f"c2_n__{f}__s{sd}__{part}.npy").astype(np.float64) + 0.5 * lgb,
                                                    np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64),
                                                    np.load(P / f"c0_clf__{f}__s{sd}__{part}.npy").astype(np.float64),
                                                    mt["s"], mt["in_cell"]) for sd in (42, 7, 2025)], axis=0)
            meta[(f, part)] = mt
    res = {"base": "V5a (0.5·c2_n + 0.5·LightGBM, compuerta y celda V4b, variante S), 3 semillas", "variants": []}
    for col, T, role in VARIANTS:
        errs, blocks, parts = {}, {}, {}
        for f in W:
            rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM[f])), threshold=T, col=col)
            for part in ("outer", "inner"):
                mt = meta[(f, part)]; y = mt["y"].astype(np.float64)
                xv = pl.DataFrame({"MVT_ID_mvt": mt["ids"]}).join(F.select("MVT_ID_mvt", col), on="MVT_ID_mvt", how="left")[col]
                xv = xv.cast(pl.Float64).to_numpy()
                pr = base_pred[(f, part)]; pr2 = apply_lobt_rule(pr, xv, rule)
                e0 = (pr - y) ** 2; e1 = (pr2 - y) ** 2
                d = e1 - e0; o = np.argsort(-np.abs(d)); keep = np.ones(d.size, bool); keep[o[:3]] = False
                parts[f"{part}-{f}"] = {"rows": int((np.isfinite(xv) & (xv > T)).sum()), "fit_rows": rule["n"],
                                        "a": rule["a"], "c": rule["c"],
                                        "delta": float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                                        "delta_without_top3": float(np.sqrt(e1[keep].mean()) - np.sqrt(e0[keep].mean()))}
                if part == "outer":
                    errs[f] = {"base": e0, "rule": e1}
                    blocks[f] = np.char.add(np.char.add(mt["airport"].astype(str), "|"), mt["day"].astype(str))
        rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in ("base", "rule")}
        bs = weighted_block_bootstrap(errs, blocks, 1000, 42); dd = bs["rule"] - bs["base"]
        res["variants"].append({"column": col, "threshold_sec": T, "role": role, "RMSE_w": rw,
                                "delta_RMSE_w": rw["rule"] - rw["base"],
                                "bootstrap_delta_ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))],
                                "parts": parts})
        v = res["variants"][-1]
        print(f"{col} > {T / 3600:.1f} h ({role}): ΔRMSE_w {v['delta_RMSE_w']:+.2f} s, IC95 [{v['bootstrap_delta_ci95'][0]:+.2f}, "
              f"{v['bootstrap_delta_ci95'][1]:+.2f}] | " + " | ".join(f"{k} {p['delta']:+.2f} ({p['delta_without_top3']:+.2f})"
                                                              for k, p in parts.items()), flush=True)
    out = OUTPUTS_DIR / "experiments" / "v6" / "rule_check.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
