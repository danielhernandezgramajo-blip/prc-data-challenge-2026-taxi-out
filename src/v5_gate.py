"""
V5 (exploración) — Compuerta de clase 1 con LightGBM, para mezclarla con la de V4b (c1_clf, CatBoost).

Mismos 9 meses de entrenamiento por fold que V4b; mismas features V4b; negativos submuestreados al 30 % con peso 1/0.3 (como
GateClassifier). Categóricas de cardinalidad alta con la tasa de clase 1 fuera de mes (k = 20). Hiperparámetros fijos.

    python src/v5_gate.py --fold LOJO --seed 42 [--outer]

Salida: models/experiments/v5/preds/lgb_gate__{fold}__s{seed}__{inner,outer}.npy
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

from config import ID_COL, MODELS_DIR  # noqa: E402
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from v2_protocol import STRATEGY, regime_class_expr  # noqa: E402
from v4b_features import load_v4b, num_cols  # noqa: E402
from v4b_protocol import GATE_NEG_FRAC, NCPU, split_fold  # noqa: E402
from v5_lgb import NATIVE_CAT, OUT_DIR, PRED_DIR, OutOfMonthTE, te_cols  # noqa: E402

GATE_LGB = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, cat_smooth=10.0, max_cat_threshold=64,
                force_col_wise=True, verbose=-1)
GATE_IT = 1000


class LGBGate:
    def __init__(self, seed: int, iterations: int = GATE_IT):
        self.seed, self.iterations = seed, iterations
        self.params = {**GATE_LGB, "seed": seed, "num_threads": NCPU}
        self.num, self.native, self.te = num_cols(), NATIVE_CAT, te_cols()

    def _matrix(self, df: pl.DataFrame, te_block: np.ndarray) -> np.ndarray:
        Xn = df.select(self.num).to_numpy().astype(np.float32)
        Xc = self.codec.transform(df.select(self.native)).to_numpy().astype(np.float32)
        return np.hstack([Xn, te_block, Xc])

    def fit(self, train: pl.DataFrame) -> "LGBGate":
        import lightgbm as lgb
        lab = (train.select(regime_class_expr())["regime_class"].to_numpy() == 1)
        rng = np.random.default_rng(self.seed)
        keep = lab | (rng.random(lab.size) < GATE_NEG_FRAC)
        # codificación fuera de mes con TODAS las filas de entrenamiento (antes de submuestrear), tasa de clase 1
        self.tenc = OutOfMonthTE(self.te)
        te_all = self.tenc.fit_transform(train, lab.astype(np.float64))
        self.codec = CategoryCodec(self.native).fit(train)
        tr = train.filter(pl.Series(keep))
        X = self._matrix(tr, te_all[keep])
        w = np.where(lab, 1.0, 1.0 / GATE_NEG_FRAC)[keep]
        names = self.num + [f"te_{c}" for c in self.te] + self.native
        cat_idx = list(range(len(self.num) + len(self.te), len(names)))
        ds = lgb.Dataset(X, label=lab[keep].astype(np.float32), weight=w, feature_name=names, categorical_feature=cat_idx)
        t0 = time.time()
        self.booster = lgb.train(self.params, ds, num_boost_round=self.iterations)
        imp = self.booster.feature_importance("gain"); imp = imp / imp.sum() * 100
        self.info = {"n_fit": int(keep.sum()), "n_pos": int(lab.sum()), "fit_seconds": round(time.time() - t0, 1),
                     "top_gain": dict(sorted(zip(names, np.round(imp, 2).tolist()), key=lambda t: -t[1])[:15])}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(self._matrix(df, self.tenc.transform(df))).astype(np.float64)


def main() -> None:
    from v2_features import RelativeCongestion
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="LOJO", choices=["LOJO", "LOJuO"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer", action="store_true")
    ap.add_argument("--iterations", type=int, default=GATE_IT)
    args = ap.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    feats = load_v4b("train")
    train, inner, outer = split_fold(feats, args.fold)
    del feats; gc.collect()
    train, _ = apply_outlier_strategy(train, STRATEGY)
    rc = RelativeCongestion().fit(train)
    train, inner, outer = rc.transform(train), rc.transform(inner), rc.transform(outer)
    for part, df in (("inner", inner), ("outer", outer)):
        meta = np.load(Path(MODELS_DIR) / "experiments" / "v4b" / "preds" / f"meta__{args.fold}__{part}.npz")
        assert np.array_equal(meta["ids"], df[ID_COL].to_numpy()), part
    t1 = time.time()
    m = LGBGate(args.seed, args.iterations).fit(train)
    np.save(PRED_DIR / f"lgb_gate__{args.fold}__s{args.seed}__inner.npy", m.predict(inner).astype(np.float32))
    if args.outer:
        np.save(PRED_DIR / f"lgb_gate__{args.fold}__s{args.seed}__outer.npy", m.predict(outer).astype(np.float32))
    rec = {"comp": "lgb_gate", "fold": args.fold, "seed": args.seed, **m.info, "seconds": round(time.time() - t1, 1)}
    with open(OUT_DIR / "units_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f"[{args.fold}] lgb_gate: {rec['seconds']:.0f}s | {list(m.info['top_gain'].items())[:8]}", flush=True)


if __name__ == "__main__":
    main()
