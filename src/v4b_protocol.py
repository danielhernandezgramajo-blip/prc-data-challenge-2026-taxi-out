"""
V4b — Componentes por (fold outer, semilla), entrenados con los 9 meses de entrenamiento y evaluados en los meses internos
(feb, ago) y en el mes outer (ene para LOJO, jul para LOJuO). Ver outputs/experiments/v4b/protocol.md.

    python src/v4b_protocol.py --seeds 42 --folds LOJuO LOJO --resume
    python src/v4b_protocol.py --seeds 7 2025 --resume

Componentes (models/experiments/v4b/preds/{comp}__{fold}__s{seed}__{part}.npy, float32; part ∈ {inner, outer}):
  c0_dn, c0_rn   rama normal de V3 (A_surface direct_normal / residual_normal; 1000 it; configuración V3)
  c0_clf         RegimeClassifier2 de V3 (modo anidado, α = 20) → (n, 3)
  c1_clf         clasificador multivariable de clase 1 (features V4b, ctr 2, negativos al 30 % con peso 1/0.3)
  c2_n           rama normal nueva (features V4b + interacciones explícitas, depth 8, 3000 it, ctr 1 — enmienda A1)
  c2_n_ctr2      ablación: la misma con max_ctr_complexity 2 (2000 it, lr 0.075; solo LOJO s42)
Metadatos por (fold, part): ids, y, aeropuerto, día, mes, MVT − SCHED, clase, celda de fecha desplazada, etc.
Anti-leakage: todo ajuste (codec, relativas, cuantiles, clasificadores, regresores) usa solo los 9 meses de entrenamiento;
las etiquetas de clase solo del bloque de entrenamiento; los meses internos y el outer solo se predicen.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, OUTLIER_HI_SEC, OUTPUTS_DIR, TARGET  # noqa: E402
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from v2_protocol import STRATEGY, make_model, regime_class_expr  # noqa: E402
from v3_protocol import DATE_SHIFT_COL, shifted_no_aobt3_expr  # noqa: E402
from v4b_features import EXP_DIR, cat_cols, load_v4b, num_cols  # noqa: E402

OUTER = {"LOJO": 1, "LOJuO": 7}
INNER = (2, 8)
SEEDS = (42, 7, 2025)
COMPONENTS = ("c2_n", "c0_dn", "c0_rn", "c0_clf", "c1_clf")
ABLATIONS = ("c2_n_ctr2",)
SMOKE = os.environ.get("V4B_SMOKE") == "1"         # prueba rápida del código: 3 % de filas, 20 iteraciones, rutas aparte
PRED_DIR = (EXP_DIR / "smoke" / "preds") if SMOKE else (EXP_DIR / "preds")
OUT_DIR = (EXP_DIR / "smoke" / "out") if SMOKE else (OUTPUTS_DIR / "experiments" / "v4b")
SMOKE_IT = 20
NCPU = os.cpu_count() or 2
GATE_NEG_FRAC = 0.30
GATE_PARAMS = dict(loss_function="Logloss", iterations=1000, learning_rate=0.08, depth=6, max_ctr_complexity=2, border_count=128,
                   l2_leaf_reg=3)
C2_PARAMS = dict(loss_function="RMSE", iterations=3000, learning_rate=0.05, depth=8, max_ctr_complexity=1, border_count=128,
                 l2_leaf_reg=3)        # enmienda A1 (solo tiempo): ctr 2 = 2.3 s/it → 1.9 h por ajuste; interacciones explícitas + ctr 1
C2_CTR2_PARAMS = dict(iterations=2000, learning_rate=0.075, max_ctr_complexity=2)   # ablación (regla de cómputo del protocolo)


def a_surface_cols() -> tuple[list[str], list[str]]:
    from v2_features import V2_CAT_COLS, V2_NUM_COLS, V2_REL_COLS
    return FEATURE_NUM_COLS + V2_NUM_COLS + V2_REL_COLS, FEATURE_CAT_COLS + V2_CAT_COLS


def month_expr() -> pl.Expr:
    return pl.col("MVT_TIME_UTC_mvt").dt.month()


def split_fold(feats: pl.DataFrame, fold: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ev = OUTER[fold]
    m = month_expr()
    assert feats["MVT_TIME_UTC_mvt"].dt.year().unique().to_list() == [2025]
    train = feats.filter(~m.is_in([ev, *INNER]))
    inner = feats.filter(m.is_in(list(INNER)))
    outer = feats.filter(m == ev)
    months_tr = sorted(train.select(m.unique())["MVT_TIME_UTC_mvt"].to_list())
    assert ev not in months_tr and not set(INNER) & set(months_tr) and len(months_tr) == 9
    return train, inner, outer


def train_months(fold: str) -> list[int]:
    return [m for m in range(1, 13) if m not in (OUTER[fold], *INNER)]


def meta_frame(df: pl.DataFrame) -> dict:
    return {"ids": df[ID_COL].to_numpy(), "y": df[TARGET].to_numpy().astype(np.float32),
            "airport": df["ADEP_mvt"].to_numpy().astype(str),
            "day": df["MVT_TIME_UTC_mvt"].dt.date().cast(pl.String).to_numpy().astype(str),
            "month": df.select(month_expr())["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int8),
            "s": df["mvt_minus_sched"].cast(pl.Float64).to_numpy().astype(np.float64),
            "regime_class": df.select(regime_class_expr())["regime_class"].to_numpy(),
            "in_cell": df.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy(),
            "has_aobt3": df["has_aobt3"].to_numpy().astype(np.int8),
            "days_after_sched": df[DATE_SHIFT_COL].to_numpy().astype(np.int16),
            "mvt_minus_aobt3": df["mvt_minus_aobt3"].cast(pl.Float64).to_numpy(),
            "hour_utc": df["hour_utc"].to_numpy().astype(np.int8)}


def pred_path(comp: str, fold: str, seed: int, part: str) -> Path:
    return PRED_DIR / f"{comp}__{fold}__s{seed}__{part}.npy"


def save_meta(fold: str, inner: pl.DataFrame, outer: pl.DataFrame) -> None:
    for part, df in (("inner", inner), ("outer", outer)):
        p = PRED_DIR / f"meta__{fold}__{part}.npz"
        if not p.exists():
            np.savez_compressed(p, **meta_frame(df))
        else:   # misma partición en todas las ejecuciones
            assert np.array_equal(np.load(p)["ids"], df[ID_COL].to_numpy()), f"ids distintos en {p}"


class GateClassifier:
    """P(clase 1 | x) con features V4b. Negativos submuestreados con peso exacto 1/frac (probabilidad sin sesgo de prior)."""

    def __init__(self, seed: int, params: dict | None = None):
        self.seed, self.params = seed, {**GATE_PARAMS, **(params or {}), **({"iterations": SMOKE_IT} if SMOKE else {})}
        self.num, self.cat = num_cols(), cat_cols()
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "GateClassifier":
        from catboost import CatBoostClassifier, Pool
        lab = (train.select(regime_class_expr())["regime_class"].to_numpy() == 1)
        rng = np.random.default_rng(self.seed)
        keep = lab | (rng.random(lab.size) < GATE_NEG_FRAC)
        w = np.where(lab, 1.0, 1.0 / GATE_NEG_FRAC)[keep]
        feats = self.num + self.cat
        self.codec = CategoryCodec(self.cat).fit(train)
        X = self.codec.transform(train.filter(pl.Series(keep)).select(feats)).to_pandas()
        t0 = time.time()
        self.clf = CatBoostClassifier(**self.params, random_seed=self.seed, thread_count=NCPU, verbose=250, allow_writing_files=False)
        self.clf.fit(Pool(X, lab[keep].astype(int), cat_features=[feats.index(c) for c in self.cat], weight=w))
        self.info = {"n_train": int(lab.size), "n_fit": int(keep.sum()), "n_pos": int(lab.sum()), "fit_seconds": round(time.time() - t0, 1),
                     "params": self.params, "feature_importance": dict(zip(feats, np.round(self.clf.get_feature_importance(), 3).tolist()))}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        feats = self.num + self.cat
        return self.clf.predict_proba(self.codec.transform(df.select(feats)).to_pandas())[:, 1]


class NormalV4b:
    """Rama normal nueva: CatBoost RMSE directo sobre clase 0 con 0 < y ≤ 3 h (features V4b)."""

    def __init__(self, seed: int, params: dict | None = None):
        self.seed, self.params = seed, {**C2_PARAMS, **(params or {}), **({"iterations": SMOKE_IT} if SMOKE else {})}
        self.num, self.cat = num_cols(), cat_cols()
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "NormalV4b":
        from catboost import CatBoostRegressor, Pool
        tr = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        feats = self.num + self.cat
        self.codec = CategoryCodec(self.cat).fit(tr)
        X = self.codec.transform(tr.select(feats)).to_pandas()
        t0 = time.time()
        self.reg = CatBoostRegressor(**self.params, random_seed=self.seed, thread_count=NCPU, verbose=250, allow_writing_files=False)
        self.reg.fit(Pool(X, tr[TARGET].to_numpy().astype(np.float32), cat_features=[feats.index(c) for c in self.cat]))
        self.info = {"n_train": tr.height, "fit_seconds": round(time.time() - t0, 1), "params": self.params,
                     "feature_importance": dict(zip(feats, np.round(self.reg.get_feature_importance(), 3).tolist()))}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        feats = self.num + self.cat
        return self.reg.predict(self.codec.transform(df.select(feats)).to_pandas()).astype(np.float64)


def fit_component(comp: str, seed: int, train: pl.DataFrame, c2_params: dict | None):
    if comp in ("c0_dn", "c0_rn"):
        num, cat = a_surface_cols()
        kind = "direct_normal" if comp == "c0_dn" else "residual_normal"
        tr = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        m = make_model(kind, num, cat, iterations=SMOKE_IT if SMOKE else 1000, params={"random_seed": seed}).fit(tr)
        info = {k: v for k, v in m.info.items() if k != "feature_importance"} | {"n_train": tr.height}
        return m.predict, info
    if comp == "c0_clf":
        import v3_protocol
        v3_protocol.SEED = seed                  # RegimeClassifier2 lee la semilla global del módulo en fit()
        m = v3_protocol.RegimeClassifier2(FEATURE_NUM_COLS, FEATURE_CAT_COLS, **({"iterations": SMOKE_IT} if SMOKE else {})).fit(train)
        info = {k: v for k, v in m.info.items() if k != "feature_importance"} | {"seed": seed}
        return (lambda df: m.predict_proba(df, mode="nested")), info
    if comp == "c1_clf":
        m = GateClassifier(seed).fit(train)
        return m.predict, m.info
    if comp == "c2_n":
        m = NormalV4b(seed, c2_params).fit(train)
        return m.predict, m.info
    if comp == "c2_n_ctr2":
        m = NormalV4b(seed, C2_CTR2_PARAMS).fit(train)
        return m.predict, m.info
    raise ValueError(comp)


def run(folds: list[str], seeds: list[int], comps: list[str], resume: bool, c2_params: dict | None, max_seconds: float | None) -> None:
    from v2_features import RelativeCongestion
    t0 = time.time()
    PRED_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = load_v4b("train")
    if SMOKE:
        feats = feats.sample(fraction=0.03, seed=0).sort("MVT_TIME_UTC_mvt")
    print(f"features V4b: {feats.shape} ({time.time() - t0:.0f}s){' [SMOKE]' if SMOKE else ''}", flush=True)
    log_path = OUT_DIR / "units_log.jsonl"
    for seed in seeds:
        for fold in folds:
            todo = [c for c in comps if not (resume and all(pred_path(c, fold, seed, p).exists() for p in ("inner", "outer")))]
            if not todo:
                print(f"[{fold} s{seed}] completo (cache)", flush=True); continue
            train, inner, outer = split_fold(feats, fold)
            train, _ = apply_outlier_strategy(train, STRATEGY)
            rc = RelativeCongestion().fit(train)
            train, inner, outer = rc.transform(train), rc.transform(inner), rc.transform(outer)
            save_meta(fold, inner, outer)
            for comp in todo:
                if max_seconds and time.time() - t0 > max_seconds:
                    print("Tanda detenida por --max-seconds; relanzar con --resume", flush=True); return
                t1 = time.time()
                print(f"[{fold} s{seed}] {comp} …", flush=True)
                predict, info = fit_component(comp, seed, train, c2_params)
                for part, df in (("inner", inner), ("outer", outer)):
                    p = np.asarray(predict(df), dtype=np.float64)
                    assert np.isfinite(p).all() and p.shape[0] == df.height
                    np.save(pred_path(comp, fold, seed, part), p.astype(np.float32))
                rec = {"fold": fold, "seed": seed, "comp": comp, "train_months": train_months(fold), "n_train_rows": train.height,
                       "seconds": round(time.time() - t1, 1), "at_unix": round(time.time()), **{k: v for k, v in info.items() if k != "feature_importance"}}
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                if "feature_importance" in info:
                    (PRED_DIR / f"{comp}__{fold}__s{seed}__importance.json").write_text(json.dumps(info["feature_importance"]), encoding="utf-8")
                print(f"[{fold} s{seed}] {comp} hecho en {rec['seconds']:.0f}s", flush=True)
                del predict; gc.collect()
            del train, inner, outer; gc.collect()
    print(f"Hecho en {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["LOJuO", "LOJO"], choices=list(OUTER))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--comps", nargs="+", default=list(COMPONENTS), choices=list(COMPONENTS) + list(ABLATIONS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--c2-iterations", type=int, default=None)
    ap.add_argument("--c2-lr", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    c2 = {k: v for k, v in (("iterations", args.c2_iterations), ("learning_rate", args.c2_lr)) if v is not None} or None
    run(args.folds, args.seeds, args.comps, args.resume, c2, args.max_seconds)


if __name__ == "__main__":
    main()
