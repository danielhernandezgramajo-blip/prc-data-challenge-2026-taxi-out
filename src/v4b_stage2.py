"""
V4b — Etapa 2: combinación rama normal + compuerta de clase 1 orientada al RMSE, validada de forma anidada (protocol.md §5–6).

    python src/v4b_stage2.py            # usa todas las (fold, semilla) con componentes completos

Para cada (fold, semilla) y configuración (C1: normal de C0; C2: normal nueva o 50/50 con la de C0), cada candidato
(normal × fuente de p × variante) se ajusta en febrero → se evalúa en agosto y viceversa (SSE interno cruzado), y se reajusta
con febrero + agosto para predecir el mes outer. Celda de fecha desplazada: mezcla de 3 clases de RegimeClassifier2 (V3) con la
normal de la configuración; fuera de la celda p2 = 0. Selección: menor SSE interno sumado sobre todas las (fold, semilla);
nunca se mira el mes outer. Salidas: models/experiments/v4b/stage2/*.npy, outputs/experiments/v4b/stage2_inner_sse.csv,
outputs/experiments/v4b/stage2_selection.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from v2_protocol import DAY_SEC  # noqa: E402
from v4b_protocol import INNER, OUT_DIR, OUTER, PRED_DIR, SEEDS, SMOKE, SMOKE_IT, pred_path  # noqa: E402

ST2_DIR = PRED_DIR.parent / "stage2"
EPS = 1e-6
PARAM_VARIANTS = ("S", "S_T", "S_R2", "S_R3", "H")
GBM_VARIANTS = ("D", "Q")
Q_MIN_ABS_DELTA = 60.0


# ---------------------------------------------------------------------------
# Piezas puras (testeables)
# ---------------------------------------------------------------------------
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def size_term(delta: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(delta) / 600.0)


def fit_platt(p: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """(a, c) de σ(a·logit p + c) por log-loss (calibración clásica)."""
    L = logit(p); y1 = y1.astype(np.float64)

    def f(th):
        z = th[0] * L + th[1]
        ll = np.logaddexp(0, z) - y1 * z
        q = sigmoid(z)
        g = q - y1
        return ll.mean(), np.array([(g * L).mean(), g.mean()])
    return minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B").x


def fit_rmse_calibration(p: np.ndarray, r: np.ndarray, delta: np.ndarray, with_size: bool) -> np.ndarray:
    """θ que minimiza Σ (r − σ(a·logit p + c [+ d·size(Δ)])·Δ)²  (r = y − b). Gradiente analítico."""
    L = logit(p); X = np.column_stack([L, np.ones_like(L)] + ([size_term(delta)] if with_size else []))
    scale = float(np.mean(delta ** 2)) + 1.0

    def f(th):
        q = sigmoid(X @ th)
        e = r - q * delta
        g = -2.0 * (e * delta * q * (1 - q)) @ X
        return float((e ** 2).sum()) / scale, g / scale
    th0 = np.array([1.0, 0.0] + ([0.0] if with_size else []))
    return minimize(f, th0, jac=True, method="L-BFGS-B").x


def apply_calibration(th: np.ndarray, p: np.ndarray, delta: np.ndarray) -> np.ndarray:
    L = logit(p)
    z = th[0] * L + th[1] + (th[2] * size_term(delta) if th.size == 3 else 0.0)
    return sigmoid(z)


def fit_hard_threshold(p: np.ndarray, y: np.ndarray, b: np.ndarray, s: np.ndarray) -> float:
    """Umbral t de la compuerta dura (ŷ = s si p > t, si no b) que minimiza el SSE. t = 1.01 ⇒ nunca."""
    order = np.argsort(-p, kind="stable")
    es = (y - s)[order] ** 2; eb = (y - b)[order] ** 2
    # SSE si las k filas con mayor p van a s: Σ es[:k] + Σ eb[k:]
    sse = np.concatenate([[0.0], np.cumsum(es)]) + (eb.sum() - np.concatenate([[0.0], np.cumsum(eb)]))
    k = int(np.argmin(sse))
    if k == 0:
        return 1.01
    ps = p[order]
    return float(ps[k - 1]) - 1e-12 if k == p.size else float((ps[k - 1] + ps[k]) / 2)


def soft(b: np.ndarray, q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return b + q * delta


def cell_mixture(P: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return P[:, 0] * b + P[:, 1] * s + P[:, 2] * (DAY_SEC + b)


# ---------------------------------------------------------------------------
# Datos por (fold, semilla)
# ---------------------------------------------------------------------------
def available(fold: str, seed: int) -> bool:
    comps = ("c0_dn", "c0_rn", "c0_clf", "c1_clf", "c2_n")
    return all(pred_path(c, fold, seed, part).exists() for c in comps for part in ("inner", "outer"))


def load_part(fold: str, seed: int, part: str) -> dict:
    meta = dict(np.load(PRED_DIR / f"meta__{fold}__{part}.npz", allow_pickle=False))
    P = np.load(pred_path("c0_clf", fold, seed, part)).astype(np.float64)
    b0 = 0.5 * np.load(pred_path("c0_dn", fold, seed, part)).astype(np.float64) + 0.5 * np.load(pred_path("c0_rn", fold, seed, part)).astype(np.float64)
    b2 = np.load(pred_path("c2_n", fold, seed, part)).astype(np.float64)
    return {**meta, "y": meta["y"].astype(np.float64), "P0": P, "b0": b0, "b2": b2,
            "p1n": np.load(pred_path("c1_clf", fold, seed, part)).astype(np.float64)}


def c0_prediction(d: dict) -> np.ndarray:
    return np.maximum(cell_mixture(d["P0"], d["b0"], d["s"]), 0.0)


def normals(d: dict, config: str) -> dict[str, np.ndarray]:
    if config == "C1":
        return {"n0": d["b0"]}
    return {"n2": d["b2"], "nb": 0.5 * (d["b2"] + d["b0"])}


def gbm_features(d: dict, idx: np.ndarray, b: np.ndarray, p: np.ndarray):
    import pandas as pd
    delta = d["s"][idx] - b[idx]
    return pd.DataFrame({"L": logit(p[idx]), "s": d["s"][idx], "b": b[idx], "delta": delta, "abs_delta": np.abs(delta),
                         "has_aobt3": d["has_aobt3"][idx].astype(np.float64), "das": d["days_after_sched"][idx].astype(np.float64),
                         "mvt_minus_aobt3": d["mvt_minus_aobt3"][idx], "hour": d["hour_utc"][idx].astype(np.float64),
                         "airport": d["airport"][idx]})


class Candidate:
    """Ajuste en `fit_idx` (de los datos internos) y predicción en filas arbitrarias (fuera de la celda)."""

    def __init__(self, variant: str, seed: int):
        self.variant, self.seed = variant, seed

    def fit(self, d: dict, idx: np.ndarray, b: np.ndarray, p: np.ndarray) -> "Candidate":
        y, s = d["y"][idx], d["s"][idx]; bi, pi = b[idx], p[idx]; delta = s - bi; r = y - bi
        v = self.variant
        if v == "S":
            pass
        elif v == "S_T":
            self.th = fit_platt(pi, d["regime_class"][idx] == 1)
        elif v in ("S_R2", "S_R3"):
            self.th = fit_rmse_calibration(pi, r, delta, with_size=(v == "S_R3"))
        elif v == "H":
            self.t = fit_hard_threshold(pi, y, bi, s)
        elif v in ("D", "Q"):
            from catboost import CatBoostRegressor, Pool
            X = gbm_features(d, idx, b, p)
            kw = dict(loss_function="RMSE", iterations=SMOKE_IT if SMOKE else 300, depth=6, learning_rate=0.1, random_seed=self.seed, verbose=0,
                      allow_writing_files=False, thread_count=os.cpu_count() or 2)
            if v == "D":
                self.m = CatBoostRegressor(**kw).fit(Pool(X, r, cat_features=["airport"]))
            else:
                ok = np.abs(delta) >= Q_MIN_ABS_DELTA
                t = np.clip(r[ok] / delta[ok], -1.0, 2.0)
                self.m = CatBoostRegressor(**kw).fit(Pool(X[ok], t, cat_features=["airport"], weight=delta[ok] ** 2))
        else:
            raise ValueError(v)
        return self

    def predict(self, d: dict, idx: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        s = d["s"][idx]; bi, pi = b[idx], p[idx]; delta = s - bi
        v = self.variant
        if v == "S":
            return soft(bi, pi, delta)
        if v == "S_T":
            return soft(bi, apply_calibration(self.th, pi, delta), delta)
        if v in ("S_R2", "S_R3"):
            return soft(bi, apply_calibration(self.th, pi, delta), delta)
        if v == "H":
            return np.where(pi > self.t, s, bi)
        X = gbm_features(d, idx, b, p)
        if v == "D":
            return bi + self.m.predict(X)
        q = np.where(np.abs(delta) >= Q_MIN_ABS_DELTA, np.clip(self.m.predict(X), 0.0, 1.0), pi)
        return soft(bi, q, delta)


def candidates_for(config: str) -> list[tuple[str, str, str]]:
    out = []
    for nk in (("n0",) if config == "C1" else ("n2", "nb")):
        for src in ("p0", "pn"):
            for v in PARAM_VARIANTS + (GBM_VARIANTS if src == "pn" else ()):
                out.append((nk, src, v))
    return out


def predict_full(d: dict, cand: Candidate, b: np.ndarray, p: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Predicción para `rows` (índices): celda → mezcla de 3 clases de V3; resto → candidato. Recorte a ≥ 0."""
    out = np.empty(rows.size)
    cell = d["in_cell"][rows]
    if cell.any():
        rc = rows[cell]
        out[cell] = cell_mixture(d["P0"][rc], b[rc], d["s"][rc])
    if (~cell).any():
        out[~cell] = cand.predict(d, rows[~cell], b, p)
    return np.maximum(out, 0.0)


def fit_rows(d: dict) -> np.ndarray:
    """Filas válidas para ajustar la etapa 2: fuera de la celda de fecha desplazada y sin clase 2."""
    return (~d["in_cell"]) & (d["regime_class"] != 2)


def crossfit_inner(d: dict, variant: str, seed: int, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Predicción interna cruzada: se ajusta en un mes interno y se predice el otro (y viceversa). La predicción de un mes
    nunca usa su propio target."""
    pred = np.empty(d["y"].size)
    ok = fit_rows(d)
    for m_fit, m_eval in ((INNER[0], INNER[1]), (INNER[1], INNER[0])):
        fit_idx = np.where(ok & (d["month"] == m_fit))[0]
        ev_idx = np.where(d["month"] == m_eval)[0]
        cand = Candidate(variant, seed).fit(d, fit_idx, b, p)
        pred[ev_idx] = predict_full(d, cand, b, p, ev_idx)
    return pred


def run_pipeline(fold: str, seed: int) -> list[dict]:
    di, do = load_part(fold, seed, "inner"), load_part(fold, seed, "outer")
    ST2_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    np.save(ST2_DIR / f"C0__{fold}__s{seed}__outer.npy", c0_prediction(do).astype(np.float32))
    inner_c0 = c0_prediction(di)
    rows.append({"fold": fold, "seed": seed, "config": "C0", "normal": "n0", "source": "p0", "variant": "V3",
                 "inner_sse": float(((inner_c0 - di["y"]) ** 2).sum()), "n_inner": int(di["y"].size)})
    fit_ok_i = fit_rows(di)
    for config in ("C1", "C2"):
        nb_i, nb_o = normals(di, config), normals(do, config)
        for nk, src, v in candidates_for(config):
            b_i, b_o = nb_i[nk], nb_o[nk]
            p_i = di["P0"][:, 1] if src == "p0" else di["p1n"]
            p_o = do["P0"][:, 1] if src == "p0" else do["p1n"]
            pred_inner = crossfit_inner(di, v, seed, b_i, p_i)
            cand = Candidate(v, seed).fit(di, np.where(fit_ok_i)[0], b_i, p_i)
            pred_outer = predict_full(do, cand, b_o, p_o, np.arange(do["y"].size))
            key = f"{config}__{nk}__{src}__{v}"
            np.save(ST2_DIR / f"{key}__{fold}__s{seed}__outer.npy", pred_outer.astype(np.float32))
            info = {}
            if hasattr(cand, "th"):
                info["theta"] = json.dumps([round(float(x), 6) for x in cand.th])
            if hasattr(cand, "t"):
                info["threshold"] = float(cand.t)
            rows.append({"fold": fold, "seed": seed, "config": config, "normal": nk, "source": src, "variant": v,
                         "inner_sse": float(((pred_inner - di["y"]) ** 2).sum()), "n_inner": int(di["y"].size), **info})
    return rows


def select(table: pl.DataFrame) -> dict:
    """Candidato primario por configuración = menor SSE interno sumado sobre todas las (fold, semilla) disponibles."""
    out = {}
    pairs = table.select("fold", "seed").unique().height
    for config in ("C1", "C2"):
        t = (table.filter(pl.col("config") == config).group_by("normal", "source", "variant")
                  .agg(pl.col("inner_sse").sum().alias("sse"), pl.len().alias("n")).filter(pl.col("n") == pairs).sort("sse"))
        r = t.row(0, named=True)
        out[config] = {"key": f"{config}__{r['normal']}__{r['source']}__{r['variant']}", "normal": r["normal"], "source": r["source"],
                       "variant": r["variant"], "inner_sse_sum": r["sse"],
                       "ranking": [{"key": f"{config}__{x['normal']}__{x['source']}__{x['variant']}", "inner_sse_sum": x["sse"]} for x in t.iter_rows(named=True)]}
    out["pairs_used"] = sorted([f"{f}_s{s}" for f, s in table.select("fold", "seed").unique().iter_rows()])
    return out


def main() -> None:
    t0 = time.time()
    pairs = [(f, s) for s in SEEDS for f in OUTER if available(f, s)]
    print("pares disponibles:", pairs, flush=True)
    all_rows = []
    for fold, seed in pairs:
        t1 = time.time()
        all_rows += run_pipeline(fold, seed)
        print(f"  {fold} s{seed}: {time.time() - t1:.0f}s", flush=True)
    table = pl.from_dicts(all_rows, infer_schema_length=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_DIR / "stage2_inner_sse.csv")
    sel = select(table)
    (OUT_DIR / "stage2_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["key"] if isinstance(v, dict) else v) for k, v in sel.items()}, indent=1), flush=True)
    print(f"Hecho en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
