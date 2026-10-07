"""
V7b — Corrección de la rama normal anclada en la congestión de los vecinos (outputs/experiments/v7/protocol.md, V7b).

En filas con anomalía nba > 600 s (fuera de la celda, sin regla LOBT): b' = b + c + w·(nb − b), con nb = media de
MVT − AOBT_3 de las demás salidas del aeropuerto en ±60 min. (c, w) por mínimos cuadrados en clase 0 (y ≤ 3 h).

    python src/v7b_neighbour.py            # validación → outputs/experiments/v7/neighbour_check.json
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

from config import TARGET  # noqa: E402
from v7_congestion import Y_MAX_FIT, airport_baseline, anomaly, apply_correction, fit_correction, neighbour_congestion, target_mask  # noqa: E402

THRESHOLD = 600.0
VARIANTS = (("nb", THRESHOLD, "principal"), ("nb", 300.0, "sensibilidad"), ("nb", 900.0, "sensibilidad"),
            ("bias", THRESHOLD, "solo sesgo"))


def main() -> None:
    from config import MODELS_DIR, OUTPUTS_DIR
    from v4b_analysis import W, weighted_block_bootstrap
    from v5_eval_inner import combine
    from v6_final import apply_lobt_rule, fit_lobt_rule
    P = MODELS_DIR / "experiments" / "v4b" / "preds"
    V5 = MODELS_DIR / "experiments" / "v5" / "preds"
    cols = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", TARGET, "mvt_minus_aobt3", "mvt_minus_lobt"]
    F = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet", columns=cols)
    nb = neighbour_congestion(F); base = airport_baseline(F, nb)
    F = F.with_columns(pl.Series("nb", nb), pl.Series("nba", anomaly(F, nb, base)), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    TRM = {"LOJO": [3, 4, 5, 6, 7, 9, 10, 11, 12], "LOJuO": [1, 3, 4, 5, 6, 9, 10, 11, 12]}
    SEEDS = (42, 7, 2025)

    def load(f: str, part: str) -> dict:
        mt = np.load(P / f"meta__{f}__{part}.npz", allow_pickle=True)
        ev = pl.DataFrame({"MVT_ID_mvt": mt["ids"]}).join(F, on="MVT_ID_mvt", how="left")
        assert ev.height == mt["ids"].size
        lgb = np.load(V5 / f"lgb_const__{f}__s42__{part}.npy").astype(np.float64)
        return {"airport": mt["airport"], "day": mt["day"], "month": mt["month"],
                "y": mt["y"].astype(np.float64), "s": mt["s"], "cell": mt["in_cell"], "rc": mt["regime_class"],
                "A": ev["mvt_minus_aobt3"].cast(pl.Float64).to_numpy(), "L": ev["mvt_minus_lobt"].cast(pl.Float64).to_numpy(),
                "nb": ev["nb"].to_numpy(), "nba": ev["nba"].to_numpy(),
                "b": [0.5 * np.load(P / f"c2_n__{f}__s{sd}__{part}.npy").astype(np.float64) + 0.5 * lgb for sd in SEEDS],
                "p1": [np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS],
                "P0": [np.load(P / f"c0_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS]}

    def sub(d: dict, keep: np.ndarray) -> dict:
        return {k: (v[keep] if isinstance(v, np.ndarray) else [x[keep] for x in v]) for k, v in d.items()}

    def predict(d: dict, rule: dict, mask: np.ndarray | None, corr: dict | None) -> np.ndarray:
        out = []
        for b, p1, P0 in zip(d["b"], d["p1"], d["P0"]):
            bb = apply_correction(b, d["nb"], mask, corr) if corr is not None else b
            out.append(combine(bb, p1, P0, d["s"], d["cell"]))
        return apply_lobt_rule(np.mean(out, axis=0), d["L"], rule)

    def check(fit_d: dict, ev_d: dict, rule: dict, thr: float, bias_only: bool) -> tuple[dict, np.ndarray, np.ndarray]:
        mi = target_mask(fit_d["nba"], fit_d["A"], fit_d["L"], fit_d["cell"], thr, need_a=False)
        corr = fit_correction(np.mean(fit_d["b"], axis=0), fit_d["y"], fit_d["nb"], mi,
                              (fit_d["rc"] == 0) & (fit_d["y"] <= Y_MAX_FIT), bias_only=bias_only)
        mo = target_mask(ev_d["nba"], ev_d["A"], ev_d["L"], ev_d["cell"], thr, need_a=False)
        p0 = predict(ev_d, rule, None, None); p1_ = predict(ev_d, rule, mo, corr)
        e0 = (p0 - ev_d["y"]) ** 2; e1 = (p1_ - ev_d["y"]) ** 2
        info = {"fit": corr, "rows_changed": int(mo.sum()), "rmse_base": float(np.sqrt(e0.mean())), "rmse_cand": float(np.sqrt(e1.mean())),
                "delta": float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                "rmse_affected_base": float(np.sqrt(e0[mo].mean())) if mo.any() else None,
                "rmse_affected_cand": float(np.sqrt(e1[mo].mean())) if mo.any() else None,
                "mean_shift_affected": float(np.mean(p1_[mo] - p0[mo])) if mo.any() else None}
        return info, e0, e1

    data = {(f, part): load(f, part) for f in W for part in ("inner", "outer")}
    res = {"variants": []}
    for anchor, thr, role in VARIANTS:
        bias_only = anchor == "bias"
        errs, blocks, checks = {}, {}, {}
        for f in W:
            rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM[f])))
            info, e0, e1 = check(data[(f, "inner")], data[(f, "outer")], rule, thr, bias_only)
            checks[f"{f}: feb+ago→outer"] = info
            errs[f] = {"base": e0, "cand": e1}
            blocks[f] = np.char.add(np.char.add(data[(f, "outer")]["airport"].astype(str), "|"), data[(f, "outer")]["day"].astype(str))
        rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM["LOJO"])))
        di = data[("LOJO", "inner")]
        info, _, _ = check(data[("LOJO", "outer")], sub(di, di["month"] == 2), rule, thr, bias_only)
        checks["LOJO: enero→febrero"] = info
        rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in ("base", "cand")}
        bs = weighted_block_bootstrap(errs, blocks, 1000, 42); dd = bs["cand"] - bs["base"]
        v = {"anchor": anchor, "threshold": thr, "role": role, "RMSE_w": rw, "delta_RMSE_w": rw["cand"] - rw["base"],
             "bootstrap_delta_ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))], "checks": checks}
        counted = [c for c in checks.values() if c["rows_changed"] >= 30]
        v["checks_counted"] = len(counted)
        v["pass"] = bool(len(counted) >= 2 and all(c["delta"] < 0 and c["rmse_affected_cand"] < c["rmse_affected_base"] for c in counted)
                         and v["bootstrap_delta_ci95"][1] < 0)
        res["variants"].append(v)

        def fmt(c: dict) -> str:
            base = f"filas {c['rows_changed']} Δ {c['delta']:+.3f}"
            if c["rows_changed"]:
                base += f" afect {c['rmse_affected_base']:.0f}→{c['rmse_affected_cand']:.0f} (+{c['mean_shift_affected']:.0f} s)"
            return base + f" [c {c['fit']['c']:+.0f}, w {c['fit']['w']:.2f}, n {c['fit']['n']}]"
        print(f"{anchor} nba>{thr:.0f} ({role}): ΔRMSE_w {v['delta_RMSE_w']:+.3f} IC95 [{v['bootstrap_delta_ci95'][0]:+.3f}, "
              f"{v['bootstrap_delta_ci95'][1]:+.3f}] | " + " | ".join(f"{k}: {fmt(c)}" for k, c in checks.items()) + f" | pass={v['pass']}", flush=True)
    res["pass_primary"] = res["variants"][0]["pass"]
    (OUTPUTS_DIR / "experiments" / "v7" / "neighbour_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
