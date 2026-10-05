"""
V6 §2 — Compuerta por prefijo de aerolínea para las salidas de LIRF sin plan NM (EOBT y LOBT nulos, fuera de la celda,
MVT − SCHED > 1800 s). Regresión logística L2 (C = 1) con dummies de banda de s y de prefijo OACI de FLIGHT_mvt
(< 5 filas → OTRO). En la candidata: p1 = 0.5·p1_V4b + 0.5·p1_prefijo en esas filas. Protocolo:
outputs/experiments/v6/protocol.md §2.

    python src/v6_prefix_gate.py          # validación (LOJO/LOJuO, outer + inner) → outputs/experiments/v6/prefix_gate_check.json
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

AIRPORT = "LIRF"
S_MIN = 1800.0
MIN_COUNT = 5
C_REG = 1.0
BLEND = 0.5
SCHED_TOL = 10.0
BANDS = [3600.0, 7200.0, 4 * 3600.0, 8 * 3600.0, 16 * 3600.0]
NEED = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "mvt_minus_sched", "mvt_minus_eobt1", "mvt_minus_lobt", "has_aobt3",
        "days_after_sched"]


def prefix_expr() -> pl.Expr:
    f = pl.col("FLIGHT_mvt").cast(pl.Utf8).fill_null("")
    return pl.coalesce(f.str.extract(r"^([A-Z]{3})", 1), f.str.slice(0, 2)).fill_null("").alias("prefix")


def target_rows_expr() -> pl.Expr:
    """Filas a las que se aplica: LIRF, sin EOBT ni LOBT, fuera de la celda (fecha desplazada sin AOBT_3), s > 1800 s."""
    in_cell = (pl.col("days_after_sched") >= 1) & (pl.col("has_aobt3").cast(pl.Int64) == 0)
    return ((pl.col("ADEP_mvt") == AIRPORT) & pl.col("mvt_minus_eobt1").is_null() & pl.col("mvt_minus_lobt").is_null()
            & ~in_cell.fill_null(False) & (pl.col("mvt_minus_sched").cast(pl.Float64) > S_MIN)).alias("pg_rows")


class PrefixGate:
    def fit(self, df: pl.DataFrame) -> "PrefixGate":
        from sklearn.linear_model import LogisticRegression
        d = df.filter(target_rows_expr()).with_columns(prefix_expr())
        y = ((d[TARGET].cast(pl.Float64) - d["mvt_minus_sched"].cast(pl.Float64)).abs() <= SCHED_TOL).to_numpy().astype(int)
        cnt = d.group_by("prefix").len()
        self.prefixes = sorted(cnt.filter(pl.col("len") >= MIN_COUNT)["prefix"].to_list())
        X = self._X(d)
        self.clf = LogisticRegression(C=C_REG, max_iter=2000).fit(X, y)
        self.n, self.pos_rate = int(d.height), float(y.mean())
        return self

    def _X(self, d: pl.DataFrame) -> np.ndarray:
        s = d["mvt_minus_sched"].cast(pl.Float64).to_numpy()
        band = np.digitize(s, BANDS)                                  # 0..5
        B = np.eye(len(BANDS) + 1)[band]
        pref = d["prefix"].to_list()
        idx = {p: i for i, p in enumerate(self.prefixes)}
        P = np.zeros((d.height, len(self.prefixes) + 1))
        for r, p in enumerate(pref):
            P[r, idx.get(p, len(self.prefixes))] = 1.0
        return np.hstack([B, P])

    def predict(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (máscara de filas afectadas, p1_prefijo en esas filas; NaN fuera)."""
        d = df.with_columns(target_rows_expr(), prefix_expr())
        mask = d["pg_rows"].to_numpy()
        out = np.full(d.height, np.nan)
        if mask.any():
            out[mask] = self.clf.predict_proba(self._X(d.filter(pl.col("pg_rows"))))[:, 1]
        return mask, out


def blend_p1(p1: np.ndarray, mask: np.ndarray, p_pref: np.ndarray) -> np.ndarray:
    out = p1.copy()
    out[mask] = BLEND * p1[mask] + (1.0 - BLEND) * p_pref[mask]
    return out


def main() -> None:
    from config import MODELS_DIR, OUTPUTS_DIR
    from v4b_analysis import W, weighted_block_bootstrap
    from v5_eval_inner import combine
    from v6_final import LOBT_GAP_SEC, apply_lobt_rule, fit_lobt_rule
    P = MODELS_DIR / "experiments" / "v4b" / "preds"
    V5 = MODELS_DIR / "experiments" / "v5" / "preds"
    F = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet",
                        columns=NEED + ["MVT_TIME_UTC_mvt", TARGET]).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    TRM = {"LOJO": [3, 4, 5, 6, 7, 9, 10, 11, 12], "LOJuO": [1, 3, 4, 5, 6, 9, 10, 11, 12]}
    res, errs, blocks = {"by_part": {}}, {}, {}
    for f in W:
        tr = F.filter(pl.col("m").is_in(TRM[f]))
        gate = PrefixGate().fit(tr)
        rule = fit_lobt_rule(tr)
        for part in ("outer", "inner"):
            mt = np.load(P / f"meta__{f}__{part}.npz", allow_pickle=True)
            y = mt["y"].astype(np.float64); s = mt["s"]; cell = mt["in_cell"]
            ev = pl.DataFrame({"MVT_ID_mvt": mt["ids"]}).join(F, on="MVT_ID_mvt", how="left")
            assert ev.height == y.size and np.allclose(ev[TARGET].cast(pl.Float64).to_numpy(), y)
            mask, pp = gate.predict(ev)
            L = ev["mvt_minus_lobt"].cast(pl.Float64).to_numpy()
            lgb = np.load(V5 / f"lgb_const__{f}__s42__{part}.npy").astype(np.float64)
            base, cand = [], []
            for sd in (42, 7, 2025):
                b = 0.5 * np.load(P / f"c2_n__{f}__s{sd}__{part}.npy").astype(np.float64) + 0.5 * lgb
                p1 = np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64)
                P0 = np.load(P / f"c0_clf__{f}__s{sd}__{part}.npy").astype(np.float64)
                base.append(apply_lobt_rule(combine(b, p1, P0, s, cell), L, rule))
                cand.append(apply_lobt_rule(combine(b, blend_p1(p1, mask, pp), P0, s, cell), L, rule))
            e0 = (np.mean(base, 0) - y) ** 2; e1 = (np.mean(cand, 0) - y) ** 2
            d = e1 - e0; o = np.argsort(-np.abs(d)); keep = np.ones(d.size, bool); keep[o[:3]] = False
            key = f"{part}-{f}"
            res["by_part"][key] = {"rows_changed": int(mask.sum()), "fit_rows": gate.n, "fit_pos_rate": round(gate.pos_rate, 3),
                                   "rmse_base": float(np.sqrt(e0.mean())), "rmse_cand": float(np.sqrt(e1.mean())),
                                   "delta": float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                                   "delta_without_top3": float(np.sqrt(e1[keep].mean()) - np.sqrt(e0[keep].mean())),
                                   "logloss_v4b": None, "logloss_blend": None}
            if mask.any():
                yc = (np.abs(y - s) <= SCHED_TOL)[mask].astype(float)
                p_v4b = np.clip(np.mean([np.load(P / f"c1_clf__{f}__s{sd}__{part}.npy").astype(np.float64) for sd in (42, 7, 2025)], 0)[mask], 1e-6, 1 - 1e-6)
                p_bl = np.clip(BLEND * p_v4b + (1 - BLEND) * pp[mask], 1e-6, 1 - 1e-6)
                ll = lambda p: float(-np.mean(yc * np.log(p) + (1 - yc) * np.log(1 - p)))  # noqa: E731
                res["by_part"][key]["logloss_v4b"], res["by_part"][key]["logloss_blend"] = ll(p_v4b), ll(p_bl)
            if part == "outer":
                errs[f] = {"base": e0, "cand": e1}
                blocks[f] = np.char.add(np.char.add(mt["airport"].astype(str), "|"), mt["day"].astype(str))
    rw = {k: float(np.sqrt(sum(W[f] * errs[f][k].mean() for f in W))) for k in ("base", "cand")}
    bs = weighted_block_bootstrap(errs, blocks, 1000, 42)
    dd = bs["cand"] - bs["base"]
    res["RMSE_w"] = rw
    res["delta_RMSE_w"] = rw["cand"] - rw["base"]
    res["bootstrap_delta_ci95"] = [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))]
    both = res["by_part"]["outer-LOJO"]["delta"] < 0 and res["by_part"]["outer-LOJuO"]["delta"] < 0
    res["rule"] = "ΔRMSE_w ≤ −1 s y mejora en ene y jul (base: V5a + regla LOBT)"
    res["pass"] = bool(res["delta_RMSE_w"] <= -1.0 and both)
    res["config"] = {"airport": AIRPORT, "s_min": S_MIN, "min_count": MIN_COUNT, "C": C_REG, "blend": BLEND, "lobt_gap": LOBT_GAP_SEC}
    out = OUTPUTS_DIR / "experiments" / "v6" / "prefix_gate_check.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
