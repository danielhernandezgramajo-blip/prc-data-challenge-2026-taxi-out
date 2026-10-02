"""
V5 (exploración) — Rama normal con LightGBM: hojas constantes frente a hojas lineales (`linear_tree`).

Cribado en los meses internos de V4b (febrero y agosto 2025), con los mismos 9 meses de entrenamiento que los componentes de
V4b de cada fold (LOJO: mar–jul + sep–dic; LOJuO: ene + mar–jun + sep–dic). Enero y julio (outer) no se miran en el cribado;
con --outer se guarda su predicción para una evaluación final aparte (como en V4b).
Mismo filtrado que NormalV4b (clase 0, 0 < y ≤ 3 h), mismas features V4b. Hiperparámetros fijos, sin early stopping.

    python src/v5_lgb.py --fold LOJO --kinds const linear --seed 42

Categóricas: las de cardinalidad baja/media van como categóricas nativas de LightGBM; las de cardinalidad alta se codifican con
la media del target fuera de mes (cada fila de entrenamiento se codifica con los OTROS meses de entrenamiento; las filas
evaluadas, con todos los meses de entrenamiento), suavizada hacia el prior con k = 20.
Salida: models/experiments/v5/preds/lgb_{kind}__{fold}__s{seed}__inner.npy (alineado con meta__{fold}__inner.npz de V4b).
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

from config import ID_COL, MODELS_DIR, OUTLIER_HI_SEC, OUTPUTS_DIR, TARGET  # noqa: E402
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from v2_protocol import STRATEGY, regime_class_expr  # noqa: E402
from v4b_features import cat_cols, load_v4b, num_cols  # noqa: E402
from v4b_protocol import NCPU, split_fold  # noqa: E402

V5_DIR = MODELS_DIR / "experiments" / "v5"
PRED_DIR = V5_DIR / "preds"
OUT_DIR = OUTPUTS_DIR / "experiments" / "v5"
NATIVE_CAT = ["ADEP_mvt", "RUNWAY_mvt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "FLIGHT_RULE_mvt", "apt_runway",
              "AIRCRAFT_TYPE_mvt"]
TE_K = 20.0
LGB_BASE = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, cat_smooth=10.0, max_cat_threshold=64,
                force_col_wise=True, verbose=-1)
LINEAR = dict(linear_tree=True, linear_lambda=1.0)
ITERATIONS = 2000
CHECKPOINTS = (500, 1000, 1500, 2000)


def te_cols() -> list[str]:
    return [c for c in cat_cols() if c not in NATIVE_CAT]


def month_of(df: pl.DataFrame) -> np.ndarray:
    return df["MVT_TIME_UTC_mvt"].dt.month().to_numpy().astype(np.int64)


class OutOfMonthTE:
    """Media del target por categoría. Entrenamiento: fuera de mes (prior también fuera de mes). Evaluación: todos los meses."""

    def __init__(self, cols: list[str], k: float = TE_K):
        self.cols, self.k = list(cols), k

    def fit_transform(self, df: pl.DataFrame, y: np.ndarray) -> np.ndarray:
        self.codec = CategoryCodec(self.cols).fit(df)
        codes = self.codec.transform(df.select(self.cols)).to_numpy().astype(np.int64)
        mi = month_of(df); months = np.unique(mi); midx = np.searchsorted(months, mi); M = months.size
        y = y.astype(np.float64)
        y_m = np.bincount(midx, weights=y, minlength=M); n_m = np.bincount(midx, minlength=M).astype(np.float64)
        prior_oom = ((y_m.sum() - y_m) / np.maximum(n_m.sum() - n_m, 1.0))[midx]
        self.prior = float(y.mean())
        out = np.empty(codes.shape, dtype=np.float32)
        self.stats = {}
        for j, c in enumerate(self.cols):
            g = codes[:, j]; G = int(g.max()) + 1
            key = g * M + midx
            s_gm = np.bincount(key, weights=y, minlength=G * M); n_gm = np.bincount(key, minlength=G * M).astype(np.float64)
            s_g = np.bincount(g, weights=y, minlength=G); n_g = np.bincount(g, minlength=G).astype(np.float64)
            out[:, j] = (s_g[g] - s_gm[key] + self.k * prior_oom) / (n_g[g] - n_gm[key] + self.k)
            self.stats[c] = (s_g, n_g)
        return out

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        codes = self.codec.transform(df.select(self.cols)).to_numpy().astype(np.int64)
        out = np.empty(codes.shape, dtype=np.float32)
        for j, c in enumerate(self.cols):
            s_g, n_g = self.stats[c]; g = codes[:, j]; ok = g >= 0; gg = np.where(ok, g, 0)
            out[:, j] = np.where(ok, (s_g[gg] + self.k * self.prior) / (n_g[gg] + self.k), self.prior)
        return out


class LGBNormal:
    """Rama normal (clase 0, 0 < y ≤ 3 h) con LightGBM; kind ∈ {const, linear}."""

    def __init__(self, seed: int, kind: str, iterations: int = ITERATIONS, params: dict | None = None):
        assert kind in ("const", "linear")
        self.seed, self.kind, self.iterations = seed, kind, iterations
        self.params = {**LGB_BASE, **(LINEAR if kind == "linear" else {}), **(params or {}), "seed": seed, "num_threads": NCPU}
        self.num, self.native, self.te = num_cols(), NATIVE_CAT, te_cols()

    def _matrix(self, df: pl.DataFrame, te_block: np.ndarray) -> np.ndarray:
        Xn = df.select(self.num).to_numpy().astype(np.float32)
        Xn = (Xn - self.mu) / self.sd
        Xc = self.codec.transform(df.select(self.native)).to_numpy().astype(np.float32)
        return np.hstack([Xn, te_block, Xc])

    def fit(self, train: pl.DataFrame) -> "LGBNormal":
        import lightgbm as lgb
        tr = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        y = tr[TARGET].to_numpy().astype(np.float64)
        Xn = tr.select(self.num).to_numpy().astype(np.float32)
        self.mu = np.nanmean(Xn, axis=0); self.sd = np.nanstd(Xn, axis=0); self.sd[~(self.sd > 0)] = 1.0
        del Xn
        self.tenc = OutOfMonthTE(self.te)
        te_block = self.tenc.fit_transform(tr, y)
        self.codec = CategoryCodec(self.native).fit(tr)
        X = self._matrix(tr, te_block)
        self.names = self.num + [f"te_{c}" for c in self.te] + self.native
        cat_idx = list(range(len(self.num) + len(self.te), len(self.names)))
        ds = lgb.Dataset(X, label=y, feature_name=self.names, categorical_feature=cat_idx, free_raw_data=False)
        t0 = time.time()
        self.booster = lgb.train(self.params, ds, num_boost_round=self.iterations)
        self.info = {"kind": self.kind, "n_train": int(tr.height), "fit_seconds": round(time.time() - t0, 1),
                     "params": self.params, "iterations": self.iterations}
        imp = self.booster.feature_importance("gain"); imp = imp / imp.sum() * 100
        self.info["top_gain"] = dict(sorted(zip(self.names, np.round(imp, 2).tolist()), key=lambda t: -t[1])[:20])
        del X, ds; gc.collect()
        return self

    def predict(self, df: pl.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        X = self._matrix(df, self.tenc.transform(df))
        return self.booster.predict(X, num_iteration=num_iteration).astype(np.float64)


def main() -> None:
    from v2_features import RelativeCongestion
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="LOJO", choices=["LOJO", "LOJuO"])
    ap.add_argument("--kinds", nargs="+", default=["const", "linear"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--outer", action="store_true", help="guardar también la predicción del mes outer (ene/jul)")
    args = ap.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = load_v4b("train")
    train, inner, outer = split_fold(feats, args.fold)
    del feats; gc.collect()
    train, _ = apply_outlier_strategy(train, STRATEGY)
    rc = RelativeCongestion().fit(train)
    train, inner, outer = rc.transform(train), rc.transform(inner), rc.transform(outer)
    for part, df in (("inner", inner), ("outer", outer)):
        meta = np.load(Path(MODELS_DIR) / "experiments" / "v4b" / "preds" / f"meta__{args.fold}__{part}.npz")
        assert np.array_equal(meta["ids"], df[ID_COL].to_numpy()), f"orden de filas distinto al de V4b ({part})"
    print(f"[{args.fold}] train {train.height:,} | inner {inner.height:,} ({time.time() - t0:.0f}s)", flush=True)
    for kind in args.kinds:
        out = PRED_DIR / f"lgb_{kind}__{args.fold}__s{args.seed}__inner.npy"
        out_o = PRED_DIR / f"lgb_{kind}__{args.fold}__s{args.seed}__outer.npy"
        if out.exists() and (out_o.exists() or not args.outer):
            print(f"{out.name} ya existe", flush=True); continue
        t1 = time.time()
        m = LGBNormal(args.seed, kind, args.iterations).fit(train)
        ks = sorted({k for k in CHECKPOINTS if k <= args.iterations} | {args.iterations})
        preds = {k: m.predict(inner, num_iteration=k) for k in ks}
        np.save(out, preds[max(preds)].astype(np.float32))
        if args.outer:   # mes outer: se guarda para la evaluación final; no se mira en el cribado
            np.save(out_o, m.predict(outer).astype(np.float32))
        for k, p in preds.items():
            if k != max(preds):
                np.save(PRED_DIR / f"lgb_{kind}__{args.fold}__s{args.seed}__inner__it{k}.npy", p.astype(np.float32))
        rec = {"fold": args.fold, "seed": args.seed, **m.info, "seconds": round(time.time() - t1, 1)}
        with open(OUT_DIR / "units_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"[{args.fold}] lgb_{kind}: {rec['seconds']:.0f}s | top gain: {list(m.info['top_gain'].items())[:8]}", flush=True)
        del m; gc.collect()


if __name__ == "__main__":
    main()
