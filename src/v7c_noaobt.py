"""
V7c — Filas sin AOBT_3 propio en congestión: b' = max(b, nb) (outputs/experiments/v7/protocol.md, V7c).

    python src/v7c_noaobt.py               # validación → outputs/experiments/v7/noaobt_check.json
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
from v7_congestion import LOBT_GAP_SEC, Y_MAX_FIT, airport_baseline, anomaly, neighbour_congestion  # noqa: E402

THRESHOLD = 600.0


def noaobt_mask(nba: np.ndarray, has_aobt3: np.ndarray, L: np.ndarray, in_cell: np.ndarray, nb: np.ndarray,
                threshold: float = THRESHOLD) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    no_rule = ~(np.isfinite(L) & (L > LOBT_GAP_SEC))
    return (np.isfinite(nba) & (nba > threshold) & np.isfinite(nb) & (np.asarray(has_aobt3) == 0)
            & no_rule & ~np.asarray(in_cell, bool))


def apply_max_nb(b: np.ndarray, nb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(b, dtype=np.float64).copy()
    out[mask] = np.maximum(out[mask], nb[mask])
    return out


def main() -> None:
    from config import MODELS_DIR, OUTPUTS_DIR
    from v4b_analysis import W, weighted_block_bootstrap
    from v5_eval_inner import combine
    from v6_final import apply_lobt_rule, fit_lobt_rule
    P = MODELS_DIR / "experiments" / "v4b" / "preds"
    V5 = MODELS_DIR / "experiments" / "v5" / "preds"
    cols = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", TARGET, "mvt_minus_aobt3", "mvt_minus_lobt", "has_aobt3"]
    F = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet", columns=cols)
    nb = neighbour_congestion(F); base = airport_baseline(F, nb)
    F = F.with_columns(pl.Series("nb", nb), pl.Series("nba", anomaly(F, nb, base)), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    TRM = {"LOJO": [3, 4, 5, 6, 7, 9, 10, 11, 12], "LOJuO": [1, 3, 4, 5, 6, 9, 10, 11, 12]}
    SEEDS = (42, 7, 2025)

    def load(f: str, part: str) -> dict:
        mt = np.load(P / f"meta__{f}__{part}.npz", allow_pickle=True)
        ev = pl.DataFrame({"MVT_ID_mvt": mt["ids"]}).join(F, on="MVT_ID_mvt", how="left")
        lgb = np.load(V5 / f"lgb_const__{f}__s42__{part}.npy").astype(np.float64)
        return {"airport": mt["airport"], "day": mt["day"], "month": mt["month"], "y": mt["y"].astype(np.float64), "s": mt["s"],
                "cell": mt["in_cell"], "rc": mt["regime_class"], "has": ev["has_aobt3"].to_numpy(),
                "L": ev["mvt_minus_lobt"].cast(pl.Float64).to_numpy(), "nb": ev["nb"].to_numpy(), "nba": ev["nba"].to_numpy(),
                "b": [0.5 * np.load(P / f"c2_n__{f}__s{sd}__{part}.npy").astype(np.float64) + 0.5 * lgb for sd in SEEDS],
                "p1": [np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS],
                "P0": [np.load(P / f"c0_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS]}

    def predict(d: dict, rule: dict, mask: np.ndarray | None, corr: dict | None = None) -> np.ndarray:
        out = []
        for b, p1, P0 in zip(d["b"], d["p1"], d["P0"]):
            if mask is not None:
                if corr is None:
                    b = apply_max_nb(b, d["nb"], mask)
                else:
                    b = b.copy(); b[mask] = np.maximum(b[mask] + corr["c"] + corr["w"] * (d["nb"][mask] - b[mask]), 0.0)
            out.append(combine(b, p1, P0, d["s"], d["cell"]))
        return apply_lobt_rule(np.mean(out, axis=0), d["L"], rule)

    def fit(d: dict, mask: np.ndarray) -> dict:
        m = mask & (d["rc"] == 0) & (d["y"] <= Y_MAX_FIT)
        bm = np.mean(d["b"], axis=0)
        Z = np.column_stack([np.ones(m.sum()), d["nb"][m] - bm[m]])
        (c, w), *_ = np.linalg.lstsq(Z, d["y"][m] - bm[m], rcond=None)
        return {"c": float(c), "w": float(w), "n": int(m.sum())}

    data = {(f, part): load(f, part) for f in W for part in ("inner", "outer")}
    res = {"variants": []}
    for kind, thr, role in (("max_nb", THRESHOLD, "principal"), ("max_nb", 300.0, "sensibilidad"), ("max_nb", 900.0, "sensibilidad"),
                            ("ajustada", THRESHOLD, "sensibilidad")):
        errs, blocks, months = {}, {}, {}
        for f in W:
            rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM[f])))
            for part in ("outer", "inner"):
                d = data[(f, part)]
                mask = noaobt_mask(d["nba"], d["has"], d["L"], d["cell"], d["nb"], thr)
                corr = None
                if kind == "ajustada":
                    fd = data[(f, "inner" if part == "outer" else "outer")]
                    corr = fit(fd, noaobt_mask(fd["nba"], fd["has"], fd["L"], fd["cell"], fd["nb"], thr))
                p0 = predict(d, rule, None); p1_ = predict(d, rule, mask, corr)
                e0 = (p0 - d["y"]) ** 2; e1 = (p1_ - d["y"]) ** 2
                for mo in np.unique(d["month"]):
                    if part == "inner" and f == "LOJuO":
                        continue                      # feb/ago ya evaluados con LOJO inner
                    sel = d["month"] == mo
                    mm = mask & sel
                    months[int(mo)] = {"rows_changed": int(mm.sum()), "delta": float(np.sqrt(e1[sel].mean()) - np.sqrt(e0[sel].mean())),
                                       "rmse_affected_base": float(np.sqrt(e0[mm].mean())) if mm.any() else None,
                                       "rmse_affected_cand": float(np.sqrt(e1[mm].mean())) if mm.any() else None,
                                       "mean_shift": float(np.mean(p1_[mm] - p0[mm])) if mm.any() else None, "fit": corr}
                if part == "outer":
                    errs[f] = {"base": e0, "cand": e1}
                    blocks[f] = np.char.add(np.char.add(d["airport"].astype(str), "|"), d["day"].astype(str))
        rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in ("base", "cand")}
        bs = weighted_block_bootstrap(errs, blocks, 1000, 42); dd = bs["cand"] - bs["base"]
        counted = {m: v for m, v in months.items() if v["rows_changed"] >= 30}
        v = {"kind": kind, "threshold": thr, "role": role, "RMSE_w": rw, "delta_RMSE_w": rw["cand"] - rw["base"],
             "bootstrap_delta_ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))], "months": months,
             "months_counted": sorted(counted)}
        v["pass"] = bool(len(counted) >= 2 and all(c["delta"] < 0 and c["rmse_affected_cand"] < c["rmse_affected_base"] for c in counted.values())
                         and v["bootstrap_delta_ci95"][1] < 0)
        res["variants"].append(v)
        txt = " | ".join(f"mes {m}: filas {c['rows_changed']} Δ {c['delta']:+.3f}" + (f" afect {c['rmse_affected_base']:.0f}→{c['rmse_affected_cand']:.0f} (+{c['mean_shift']:.0f})" if c["rows_changed"] else "")
                         for m, c in sorted(months.items()))
        print(f"{kind} nba>{thr:.0f} ({role}): ΔRMSE_w {v['delta_RMSE_w']:+.3f} IC95 [{v['bootstrap_delta_ci95'][0]:+.3f}, {v['bootstrap_delta_ci95'][1]:+.3f}] | {txt} | pass={v['pass']}", flush=True)
    res["pass_primary"] = res["variants"][0]["pass"]
    (OUTPUTS_DIR / "experiments" / "v7" / "noaobt_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
