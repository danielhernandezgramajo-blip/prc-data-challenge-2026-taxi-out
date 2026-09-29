"""
V2 — Protocolo de validación temporal con origen creciente (8 folds) y ejecución de componentes.

Folds (todo 2025; cortes por MVT_TIME_UTC_mvt):
    May: train ene–abr → val may   ...   Nov: train ene–oct → val nov   (7 folds de DESARROLLO)
    Dec: train ene–nov → val dic                                          (HOLDOUT: se abre una sola vez)

Componentes por (variante, fold), guardados en models/experiments/v2/preds/ (float32, --resume):
    v1_protocol :: direct, residual
        Features V1, CatBoost 1000 iteraciones fijas, drop_nonpositive: exactamente la configuración del modelo
        final V1. Los backtests antiguos (jul/nov/dic) usaban early stopping → no son "exactamente compatibles"
        y se recomputan bajo este protocolo. El ensamble 50/50 de estos dos componentes es "V1 bajo protocolo".
    S_mixture :: proba (3 clases), direct_normal
        Hallazgo del análisis de cola (v2_tail_analysis): cuando el aeropuerto no registra el off-block real,
        BLOCK_TIME se rellena con SCHED_TIME y el target vale EXACTAMENTE MVT_TIME − SCHED_TIME (170 de los 202
        extremos > 3 h de 2025; 100 049 filas en total). Otra familia (17 filas) vale ≈ 86 400 s + taxi normal
        (fecha del bloque desplazada un día). Ambas son funciones conocidas de columnas disponibles en ranking.
        Clases: 0 = taxi físico; 1 = |y − (MVT − SCHED)| ≤ 10 s; 2 = y − 86 400 ∈ [0, 3 h] y no clase 1.
        proba = CatBoost binario (clase 1) SIN pesos de clase + tasa suavizada (clase 2), calibradas por logloss;
        direct_normal = CatBoost directo entrenado solo con filas de clase 0 y y ≤ 3 h.
        Predicción (en v2_analysis): E[y|x] = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal),
        con `normal` = 0.5·direct_normal + 0.5·residual(v1_protocol) (o variantes evaluadas).
    A_surface :: direct, residual  (features V1 + congestión temporal v2_features; etapa 2)

Garantías anti-leakage: todo ajuste (codec, medianas, congestión relativa, clasificador, regresores) usa solo
train < corte; las etiquetas de clase se derivan del target de ENTRENAMIENTO únicamente; la validación se
puntúa siempre con el target original y completo (extremos incluidos).

Uso:
    python src/v2_protocol.py --variants v1_protocol S_mixture --folds May Jun Jul Aug Sep Oct Nov --resume
    python src/v2_protocol.py --variants <finalista> --folds Dec --resume      # holdout, una sola vez
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import (  # noqa: E402
    CATBOOST_PARAMS, FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, MODELS_DIR, OUTLIER_HI_SEC, OUTPUTS_DIR, SEED, TARGET,
)
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from train_baseline import CatBoostDirect, CatBoostResidual  # noqa: E402
from v2_metrics import block  # noqa: E402

np.random.seed(SEED)

V2_FOLDS = {"May": (1, 4, 5), "Jun": (1, 5, 6), "Jul": (1, 6, 7), "Aug": (1, 7, 8), "Sep": (1, 8, 9),
            "Oct": (1, 9, 10), "Nov": (1, 10, 11), "Dec": (1, 11, 12)}
DEV_FOLDS = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
HOLDOUT_FOLD = "Dec"
STRATEGY = "drop_nonpositive"
ITERATIONS = 1000            # igual que el modelo final V1 (models/final/*/manifest.json)
DAY_SEC = 86_400.0
SCHED_TOL = 10.0             # |y − (MVT − SCHED)| ≤ 10 s ⇒ bloque rellenado con la hora programada
EXP_DIR = MODELS_DIR / "experiments" / "v2"
PRED_DIR = EXP_DIR / "preds"
OUT_DIR = OUTPUTS_DIR / "experiments" / "v2"


def variants() -> dict:
    from v2_features import V2_CAT_COLS, V2_NUM_COLS, V2_REL_COLS
    return {
        "v1_protocol": {"num": FEATURE_NUM_COLS, "cat": FEATURE_CAT_COLS, "kinds": ["direct", "residual"], "surface": False},
        "S_mixture": {"num": FEATURE_NUM_COLS, "cat": FEATURE_CAT_COLS, "kinds": ["proba", "direct_normal"], "surface": False},
        "A_surface": {"num": FEATURE_NUM_COLS + V2_NUM_COLS + V2_REL_COLS, "cat": FEATURE_CAT_COLS + V2_CAT_COLS,
                      "kinds": ["direct", "residual"], "surface": True},
    }


def split_fold(feats: pl.DataFrame, fold: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    m0, m1, mv = V2_FOLDS[fold]
    m = pl.col("MVT_TIME_UTC_mvt").dt.month(); yr = pl.col("MVT_TIME_UTC_mvt").dt.year()
    train = feats.filter((yr == 2025) & (m >= m0) & (m <= m1)); val = feats.filter((yr == 2025) & (m == mv))
    assert train["MVT_TIME_UTC_mvt"].max() < val["MVT_TIME_UTC_mvt"].min()
    return train, val


def regime_class_expr() -> pl.Expr:
    """Clase de régimen derivada del target (SOLO para etiquetar el bloque de entrenamiento)."""
    y = pl.col(TARGET).cast(pl.Float64); s = pl.col("mvt_minus_sched").cast(pl.Float64)
    c1 = (y - s).abs() <= SCHED_TOL
    c2 = ((y - DAY_SEC) >= 0) & ((y - DAY_SEC) <= OUTLIER_HI_SEC) & ~c1
    return pl.when(c1).then(1).when(c2).then(2).otherwise(0).cast(pl.Int8).alias("regime_class")


def make_model(kind: str, num: list[str], cat: list[str], iterations: int = ITERATIONS, params: dict | None = None):
    """CatBoost directo/residual con lista de features arbitraria (misma clase que V1)."""
    codec = CategoryCodec(cat)
    m = CatBoostDirect(codec=codec, fixed_iterations=iterations, params=params) if kind in ("direct", "direct_normal") else \
        CatBoostResidual(codec=codec, fixed_iterations=iterations, params=params)
    m.features = list(num) + list(cat)
    m.cat_idx = [m.features.index(c) for c in cat]
    orig_fit = m.fit

    def fit(train):
        m.codec.fit(train)   # fit() solo crea el codec cuando es None; aquí ya existe con las categóricas de la variante
        return orig_fit(train)
    m.fit = fit
    return m


class RegimeClassifier:
    """P(clase | x). Clase 1 (bloque = hora programada): CatBoost binario Logloss SIN reponderar clases
    (probabilidad calibrada por logloss). Clase 2 (fecha del bloque desplazada un día; 17 filas en 2025): tasa
    empírica suavizada por (aeropuerto, has_aobt3) del bloque de entrenamiento — demasiado rara para un modelo.
    Devuelve una matriz (n, 3) con p0 = 1 − p1 − p2."""
    name = "proba"

    def __init__(self, num, cat, iterations: int = 300):
        self.num, self.cat, self.iterations = list(num), list(cat), iterations
        self.info = {}

    def fit(self, train: pl.DataFrame) -> "RegimeClassifier":
        from catboost import CatBoostClassifier, Pool
        lab = train.select(regime_class_expr())["regime_class"].to_numpy()
        self.codec = CategoryCodec(self.cat).fit(train)
        feats = self.num + self.cat
        X = self.codec.transform(train.select(feats)).to_pandas()
        cat_idx = [feats.index(c) for c in self.cat]
        self.clf = CatBoostClassifier(loss_function="Logloss", iterations=self.iterations, learning_rate=0.1, depth=6,
                                      random_seed=SEED, thread_count=CATBOOST_PARAMS["thread_count"], verbose=0,
                                      allow_writing_files=False, max_ctr_complexity=1)
        t0 = time.time()
        self.clf.fit(Pool(X, (lab == 1).astype(int), cat_features=cat_idx))
        # tasa de clase 2 por (aeropuerto, has_aobt3), suavizada hacia la tasa global con pseudo-recuento de 200 filas
        g = (train.with_columns(regime_class_expr()).group_by("ADEP_mvt", "has_aobt3")
                  .agg(pl.len().alias("n"), (pl.col("regime_class") == 2).sum().alias("k")))
        glob = float((lab == 2).mean())
        self.rate2 = {(r["ADEP_mvt"], int(r["has_aobt3"])): (r["k"] + 200 * glob) / (r["n"] + 200) for r in g.iter_rows(named=True)}
        self.rate2_global = glob
        self.info = {"n_train": int(lab.size), "n_class1": int((lab == 1).sum()), "n_class2": int((lab == 2).sum()),
                     "train_seconds": round(time.time() - t0, 1)}
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        feats = self.num + self.cat
        X = self.codec.transform(df.select(feats)).to_pandas()
        p1 = self.clf.predict_proba(X)[:, 1]
        keys = list(zip(df["ADEP_mvt"].to_list(), df["has_aobt3"].cast(pl.Int64).to_list()))
        p2 = np.array([self.rate2.get(k, self.rate2_global) for k in keys], dtype=np.float64)
        p1 = np.clip(p1 * (1 - p2), 0, 1)
        p0 = np.clip(1 - p1 - p2, 0, 1)
        return np.column_stack([p0, p1, p2])


def run_unit(variant: str, fold: str, feats: pl.DataFrame, resume: bool) -> list[dict]:
    cfg = variants()[variant]
    train, val = split_fold(feats, fold)
    train, sinfo = apply_outlier_strategy(train, STRATEGY)
    if cfg["surface"]:
        from v2_features import RelativeCongestion
        rc = RelativeCongestion().fit(train)      # solo bloque de entrenamiento
        train, val = rc.transform(train), rc.transform(val)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = PRED_DIR / f"meta__{fold}.npz"
    if not meta_path.exists():
        np.savez_compressed(meta_path, y=val[TARGET].to_numpy().astype(np.float32), ids=val[ID_COL].to_numpy(),
                            airport=val["ADEP_mvt"].to_numpy().astype(str),
                            day=val["MVT_TIME_UTC_mvt"].dt.date().cast(pl.String).to_numpy().astype(str),
                            mvt_minus_sched=val["mvt_minus_sched"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float32),
                            mvt_minus_aobt3=val["mvt_minus_aobt3"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float32),
                            regime_class=val.select(regime_class_expr())["regime_class"].to_numpy())
    y = val[TARGET].to_numpy().astype(np.float64)
    rows = []
    for kind in cfg["kinds"]:
        f = PRED_DIR / f"{variant}__{fold}__{kind}.npy"
        if resume and f.exists():
            print(f"  {variant:12s} {fold} {kind:13s} (cache)", flush=True)
            continue
        t0 = time.time()
        if kind == "proba":
            m = RegimeClassifier(cfg["num"], cfg["cat"]).fit(train)
            out = m.predict_proba(val); np.save(f, out.astype(np.float32))
            lab = val.select(regime_class_expr())["regime_class"].to_numpy()
            from sklearn.metrics import log_loss, roc_auc_score
            auc1 = float(roc_auc_score(lab == 1, out[:, 1])) if (lab == 1).any() else np.nan
            auc2 = float(roc_auc_score(lab == 2, out[:, 2])) if (lab == 2).any() else np.nan
            rows.append({"variant": variant, "fold": fold, "model": kind, "n": int(y.size), "fit_seconds": round(time.time() - t0, 1),
                         "logloss": float(log_loss(lab, out, labels=[0, 1, 2])), "auc_class1": auc1, "auc_class2": auc2,
                         "mean_p1": float(out[:, 1].mean()), "obs_rate1": float((lab == 1).mean()),
                         "mean_p2": float(out[:, 2].mean()), "obs_rate2": float((lab == 2).mean()), **m.info})
            print(f"  {variant:12s} {fold} {kind:13s} logloss={rows[-1]['logloss']:.4f} auc1={auc1:.3f} auc2={auc2:.3f} "
                  f"p1={rows[-1]['mean_p1']:.4f}/{rows[-1]['obs_rate1']:.4f} fit={rows[-1]['fit_seconds']}s", flush=True)
            continue
        if kind == "direct_normal":
            tr_k = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        else:
            tr_k = train
        m = make_model(kind, cfg["num"], cfg["cat"]).fit(tr_k)
        pred = m.predict(val); assert np.isfinite(pred).all()
        np.save(f, pred.astype(np.float32))
        fi = m.info.get("feature_importance")
        if fi:
            (PRED_DIR / f"{variant}__{fold}__{kind}__importance.json").write_text(json.dumps(fi), encoding="utf-8")
        rows.append({"variant": variant, "fold": fold, "model": kind, **block(y, pred), "fit_seconds": round(time.time() - t0, 1),
                     "n_train": tr_k.height})
        print(f"  {variant:12s} {fold} {kind:13s} RMSE={rows[-1]['rmse']:7.1f} MAE={rows[-1]['mae']:6.1f} "
              f"cov5={rows[-1]['cov_le300']:5.2f}% fit={rows[-1]['fit_seconds']}s n_train={tr_k.height:,}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["v1_protocol"], choices=list(variants()))
    ap.add_argument("--folds", nargs="+", default=DEV_FOLDS, choices=list(V2_FOLDS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    if HOLDOUT_FOLD in args.folds:
        print("AVISO: se está abriendo el holdout de diciembre. Debe hacerse UNA sola vez con la variante finalista.", flush=True)
    t0 = time.time()
    if any(variants()[v]["surface"] for v in args.variants):
        from v2_features import load_features_v2
        feats = load_features_v2("train")
    else:
        from prepare_training import load_features
        feats = load_features("train")
    print(f"features: {feats.shape} ({time.time()-t0:.1f}s)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "units_log.csv"
    rows = pl.read_csv(log).to_dicts() if log.exists() else []
    for fold in args.folds:
        for variant in args.variants:
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("Tanda detenida por --max-seconds; relanzar con --resume"); return
            new = run_unit(variant, fold, feats, args.resume)
            done_kinds = {r["model"] for r in new}
            rows = [x for x in rows if not (x["variant"] == variant and x["fold"] == fold and x["model"] in done_kinds)] + new
            if rows:
                pl.from_dicts(rows, infer_schema_length=None).write_csv(log)
    print(f"Hecho en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
