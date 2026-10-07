"""
V7 — Corrección de la rama normal en congestión extrema (outputs/experiments/v7/protocol.md).

nb60 = media de MVT − AOBT_3 (en (0, 2 h)) de las DEMÁS salidas del mismo aeropuerto en ±60 min; nba = nb60 − mediana 2025
del aeropuerto. Solo features (AOBT_3 de otros vuelos), ningún target. En las filas con nba > umbral, A = MVT − AOBT_3 propio
válido, fuera de la celda y sin regla LOBT: b' = b + c + w·(A − b), con (c, w) por mínimos cuadrados en clase 0 (y ≤ 3 h).

    python src/v7_congestion.py            # validación → outputs/experiments/v7/congestion_check.json
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

WINDOW_MIN = 60
A_MAX = 7200.0
NBA_THRESHOLD = 900.0
LOBT_GAP_SEC = 7200.0
Y_MAX_FIT = 3 * 3600.0


def neighbour_congestion(df: pl.DataFrame, window_min: int = WINDOW_MIN) -> np.ndarray:
    """Media de A = MVT − AOBT_3 (solo valores en (0, 2 h)) de las otras salidas del mismo aeropuerto en ±window, sin la
    propia fila. Devuelve un array alineado con df (NaN si no hay vecinos con A válido)."""
    t_all = df["MVT_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64).to_numpy()
    a_all = df["mvt_minus_aobt3"].cast(pl.Float64).to_numpy()
    a_all = np.where((a_all > 0) & (a_all < A_MAX), a_all, np.nan)
    apt = df["ADEP_mvt"].to_numpy()
    out = np.full(df.height, np.nan)
    for ap in np.unique(apt):
        idx = np.where(apt == ap)[0]
        o = idx[np.argsort(t_all[idx], kind="stable")]
        t = t_all[o]; v = a_all[o]; has = np.isfinite(v); vv = np.where(has, v, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(vv)]); cn = np.concatenate([[0], np.cumsum(has)])
        lo = np.searchsorted(t, t - window_min * 60, "left"); hi = np.searchsorted(t, t + window_min * 60, "right")
        s_ = cs[hi] - cs[lo] - vv; c_ = cn[hi] - cn[lo] - has
        out[o] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
    return out


def airport_baseline(df: pl.DataFrame, nb: np.ndarray) -> dict[str, float]:
    """Mediana de nb60 por aeropuerto (se calcula con 2025)."""
    d = pl.DataFrame({"apt": df["ADEP_mvt"], "nb": nb}).with_columns(pl.col("nb").fill_nan(None))
    return {r[0]: float(r[1]) for r in d.group_by("apt").agg(pl.col("nb").median()).iter_rows()}


def anomaly(df: pl.DataFrame, nb: np.ndarray, base: dict[str, float]) -> np.ndarray:
    med = np.array([base.get(a, np.nan) for a in df["ADEP_mvt"].to_list()], dtype=np.float64)
    return nb - med


def target_mask(nba: np.ndarray, A: np.ndarray, L: np.ndarray, in_cell: np.ndarray, threshold: float = NBA_THRESHOLD,
                need_a: bool = True) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64); L = np.asarray(L, dtype=np.float64)
    a_ok = (np.isfinite(A) & (A > 0) & (A < A_MAX)) if need_a else np.ones(A.size, bool)
    no_rule = ~(np.isfinite(L) & (L > LOBT_GAP_SEC))
    return np.isfinite(nba) & (nba > threshold) & a_ok & no_rule & ~np.asarray(in_cell, bool)


def fit_correction(b: np.ndarray, y: np.ndarray, X: np.ndarray, mask: np.ndarray, normal: np.ndarray, bias_only: bool = False) -> dict:
    """(y − b) = c + w·(X − b) por mínimos cuadrados en las filas `mask & normal` (clase 0, y ≤ 3 h). X es el ancla:
    A propio (V7) o nb de los vecinos (V7b). Con bias_only, w = 0."""
    m = mask & normal
    if m.sum() < 30:
        raise ValueError(f"muy pocas filas para ajustar la corrección ({int(m.sum())})")
    if bias_only:
        return {"c": float(np.mean(y[m] - b[m])), "w": 0.0, "n": int(m.sum())}
    Z = np.column_stack([np.ones(m.sum()), X[m] - b[m]])
    (c, w), *_ = np.linalg.lstsq(Z, y[m] - b[m], rcond=None)
    return {"c": float(c), "w": float(w), "n": int(m.sum())}


def apply_correction(b: np.ndarray, X: np.ndarray, mask: np.ndarray, corr: dict) -> np.ndarray:
    out = np.asarray(b, dtype=np.float64).copy()
    xx = np.nan_to_num(np.asarray(X, dtype=np.float64)[mask], nan=0.0)
    out[mask] = np.maximum(out[mask] + corr["c"] + corr["w"] * (xx - out[mask]), 0.0)
    return out


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
    F = F.with_columns(pl.Series("nba", anomaly(F, nb, base)), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    TRM = {"LOJO": [3, 4, 5, 6, 7, 9, 10, 11, 12], "LOJuO": [1, 3, 4, 5, 6, 9, 10, 11, 12]}
    SEEDS = (42, 7, 2025)

    def load(f: str, part: str) -> dict:
        mt = np.load(P / f"meta__{f}__{part}.npz", allow_pickle=True)
        ev = pl.DataFrame({"MVT_ID_mvt": mt["ids"]}).join(F, on="MVT_ID_mvt", how="left")
        assert ev.height == mt["ids"].size
        lgb = np.load(V5 / f"lgb_const__{f}__s42__{part}.npy").astype(np.float64)
        bs = [0.5 * np.load(P / f"c2_n__{f}__s{sd}__{part}.npy").astype(np.float64) + 0.5 * lgb for sd in SEEDS]
        return {"mt": mt, "y": mt["y"].astype(np.float64), "s": mt["s"], "cell": mt["in_cell"], "rc": mt["regime_class"],
                "A": ev["mvt_minus_aobt3"].cast(pl.Float64).to_numpy(), "L": ev["mvt_minus_lobt"].cast(pl.Float64).to_numpy(),
                "nba": ev["nba"].to_numpy(), "b": bs,
                "p1": [np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS],
                "P0": [np.load(P / f"c0_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in SEEDS]}

    def predict(d: dict, rule: dict, mask: np.ndarray | None, corr: dict | None) -> np.ndarray:
        out = []
        for b, p1, P0 in zip(d["b"], d["p1"], d["P0"]):
            bb = apply_correction(b, d["A"], mask, corr) if corr is not None else b
            out.append(combine(bb, p1, P0, d["s"], d["cell"]))
        return apply_lobt_rule(np.mean(out, axis=0), d["L"], rule)

    res = {"config": {"window_min": WINDOW_MIN, "threshold_primary": NBA_THRESHOLD, "a_max": A_MAX}, "variants": []}
    data = {(f, part): load(f, part) for f in W for part in ("inner", "outer")}
    feb = {f: data[(f, "inner")]["mt"]["month"] == 2 for f in W}

    def sub(d: dict, keep: np.ndarray) -> dict:
        out = {k: (v[keep] if isinstance(v, np.ndarray) else [x[keep] for x in v] if isinstance(v, list) else v)
               for k, v in d.items() if k != "mt"}
        out["mt"] = {k: d["mt"][k][keep] for k in ("airport", "day")}
        return out

    def check(f: str, fit_d: dict, ev_d: dict, rule: dict, thr: float) -> tuple[dict, np.ndarray, np.ndarray]:
        mi = target_mask(fit_d["nba"], fit_d["A"], fit_d["L"], fit_d["cell"], thr)
        corr = fit_correction(np.mean(fit_d["b"], axis=0), fit_d["y"], fit_d["A"], mi, (fit_d["rc"] == 0) & (fit_d["y"] <= Y_MAX_FIT))
        mo = target_mask(ev_d["nba"], ev_d["A"], ev_d["L"], ev_d["cell"], thr)
        p0 = predict(ev_d, rule, None, None); p1_ = predict(ev_d, rule, mo, corr)
        e0 = (p0 - ev_d["y"]) ** 2; e1 = (p1_ - ev_d["y"]) ** 2
        info = {"fit": corr, "rows_changed": int(mo.sum()), "rmse_base": float(np.sqrt(e0.mean())), "rmse_cand": float(np.sqrt(e1.mean())),
                "delta": float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                "rmse_affected_base": float(np.sqrt(e0[mo].mean())) if mo.any() else None,
                "rmse_affected_cand": float(np.sqrt(e1[mo].mean())) if mo.any() else None}
        return info, e0, e1

    for thr in (NBA_THRESHOLD, 600.0, 1200.0):
        errs, blocks, checks = {}, {}, {}
        for f in W:
            rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM[f])))
            info, e0, e1 = check(f, data[(f, "inner")], data[(f, "outer")], rule, thr)
            checks[f"{f}: inner→outer"] = info
            errs[f] = {"base": e0, "cand": e1}
            blocks[f] = np.char.add(np.char.add(data[(f, "outer")]["mt"]["airport"].astype(str), "|"), data[(f, "outer")]["mt"]["day"].astype(str))
        rule = fit_lobt_rule(F.filter(pl.col("m").is_in(TRM["LOJO"])))
        info, _, _ = check("LOJO", data[("LOJO", "outer")], sub(data[("LOJO", "inner")], feb["LOJO"]), rule, thr)
        checks["LOJO: enero→febrero"] = info
        rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in ("base", "cand")}
        bs = weighted_block_bootstrap(errs, blocks, 1000, 42); dd = bs["cand"] - bs["base"]
        v = {"threshold": thr, "RMSE_w": rw, "delta_RMSE_w": rw["cand"] - rw["base"],
             "bootstrap_delta_ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))], "checks": checks}
        counted = [c for c in checks.values() if c["rows_changed"] >= 30]
        v["checks_counted"] = len(counted)
        v["pass"] = bool(len(counted) >= 2 and all(c["delta"] < 0 and c["rmse_affected_cand"] < c["rmse_affected_base"] for c in counted)
                         and v["bootstrap_delta_ci95"][1] < 0)
        res["variants"].append(v)
        fmt = lambda c: (f"filas {c['rows_changed']} Δ {c['delta']:+.3f}" + (f" afectadas {c['rmse_affected_base']:.0f}→{c['rmse_affected_cand']:.0f}"  # noqa: E731
                         if c["rows_changed"] else "") + f" (c {c['fit']['c']:+.0f}, w {c['fit']['w']:.2f}, n {c['fit']['n']})")
        print(f"nba > {thr:.0f}: ΔRMSE_w {v['delta_RMSE_w']:+.3f} s IC95 [{v['bootstrap_delta_ci95'][0]:+.3f}, {v['bootstrap_delta_ci95'][1]:+.3f}] | "
              + " | ".join(f"{k}: {fmt(c)}" for k, c in checks.items()) + f" | pass={v['pass']}", flush=True)
    res["pass_primary"] = res["variants"][0]["pass"]
    out = OUTPUTS_DIR / "experiments" / "v7" / "congestion_check.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
