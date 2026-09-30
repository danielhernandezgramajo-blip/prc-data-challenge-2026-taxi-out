"""
V3 = etapa 2 de V2 — mismos 8 folds de origen creciente (v2_protocol), dos familias nuevas de componentes.

Hallazgo que motiva la etapa 2a (análisis del error restante de S_mix en desarrollo): el 25 % del error cuadrático
que queda está en 10 filas de clase 2 (y = taxi + 86 400) y el 16 % en la clase 1. Mecanismo de la clase 2: en LIRF,
cuando no hay AOBT_3 y el vuelo sale en una FECHA posterior a la programada, el off-block se registra con la hora
correcta pero con la fecha del programado → y = taxi + 86 400. En 2025 esa celda (LIRF, sin AOBT_3, fecha MVT >
fecha SCHED) tiene 68 filas: 54 de clase 1, 11 de clase 2 y 3 normales; el clasificador de V2 les da p2 ≈ 0,005.

Componentes por (variante, fold), guardados en models/experiments/v3/preds/ (float32, --resume). Los componentes de
V2 (v1_protocol, S_mixture) se reutilizan desde models/experiments/v2/preds/ sin recalcular ni modificar.

    S2_mixture :: proba_a{α}   (α ∈ ALPHAS; UNA sola ejecución del clasificador por fold)
        RegimeClassifier2: CatBoost binario (clase 1) con features V1 + `days_after_sched` (fecha MVT − fecha SCHED,
        en días; ambas columnas están en ranking) + tabla de clase 2 por (aeropuerto, has_aobt3, desplazada) con
        pseudo-recuento 200 hacia la tasa global. En las celdas "desplazada ≥ 1 día y sin AOBT_3" (por aeropuerto)
        se sustituyen p1 y p2 por la actualización bayesiana (k + α·prior)/(n + α) con las cuentas de la celda en el
        bloque de ENTRENAMIENTO y prior = (p1 del clasificador, p2 de la tabla). α = 0 es la ablación sin celdas;
        α = 20 es el valor PRIMARIO prefijado; 5 y 100 son sensibilidad (no se selecciona entre ellos).
        proba_b20 (etapa 2c, POST-HOC: definida tras abrir diciembre, ver informe): celdas jerárquicas por banda de
        retraso dentro de la celda del aeropuerto, banda = (MVT − SCHED ≥ 12 h): p_banda = (k_banda + α·p_celda)/(n_banda + α).
        Motivo: en 2025 la clase 2 se concentra en retrasos ≥ 12 h (10 de 25 filas frente a 1 de 43 con retraso < 12 h) y la
        celda plana repartía esa masa a filas de retraso corto (visto en diciembre); el mismo ajuste del clasificador.
    A_surface :: direct_normal, residual_normal   (features V1 + congestión de v2_features; etapa 2b)
        Ambos entrenados SOLO con filas de clase 0 y y ≤ 3 h (rama "normal" de la mezcla no contaminada por la cola).
        residual_normal es el análogo de direct_normal para el residual anclado en AOBT_3 (fallback mediana jerárquica).

Garantías: mismas particiones, misma estrategia (drop_nonpositive), mismos ids/orden de validación que V2 (se
verifica contra meta__{fold}.npz de V2); etiquetas de régimen solo del bloque de entrenamiento; diciembre sellado
hasta congelar la finalista (segunda apertura, documentada como confirmación).

Uso:
    python src/v3_protocol.py --variants S2_mixture A_surface --folds May Jun Jul Aug Sep Oct Nov --resume
    python src/v3_protocol.py --variants <finalista> --folds Dec --resume
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
    CATBOOST_PARAMS, DEP, FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, MODELS_DIR, OUTLIER_HI_SEC, OUTPUTS_DIR, PHASE_COL,
    SEED, TARGET,
)
from io_utils import scan_ranking, scan_training  # noqa: E402
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from v2_metrics import block  # noqa: E402
from v2_protocol import (  # noqa: E402
    DAY_SEC, DEV_FOLDS, HOLDOUT_FOLD, ITERATIONS, STRATEGY, V2_FOLDS, make_model, regime_class_expr, split_fold,
)
from v2_protocol import PRED_DIR as PRED_DIR_V2  # noqa: E402

np.random.seed(SEED)

DATE_SHIFT_COL = "days_after_sched"
ALPHAS = (0, 5, 20, 100)
ALPHA_PRIMARY = 20
RATE2_PSEUDO = 200
BAND_SEC = 12 * 3600.0       # etapa 2c: celdas jerárquicas por banda de retraso (MVT − SCHED ≥ 12 h) dentro de la celda del aeropuerto
CELL_MODES = ("flat", "band")
EXP_DIR = MODELS_DIR / "experiments" / "v3"
PRED_DIR = EXP_DIR / "preds"
OUT_DIR = OUTPUTS_DIR / "experiments" / "v3"
_DATE_RAW = [ID_COL, PHASE_COL, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def variants() -> dict:
    from v2_features import V2_CAT_COLS, V2_NUM_COLS, V2_REL_COLS
    return {
        "S2_mixture": {"num": FEATURE_NUM_COLS, "cat": FEATURE_CAT_COLS, "kinds": ["proba"], "surface": False},
        "A_surface": {"num": FEATURE_NUM_COLS + V2_NUM_COLS + V2_REL_COLS, "cat": FEATURE_CAT_COLS + V2_CAT_COLS,
                      "kinds": ["direct_normal", "residual_normal"], "surface": True},
    }


# ---------------------------------------------------------------------------
# Feature de desplazamiento de fecha (sin target, sin BLOCK_TIME; disponible en ranking)
# ---------------------------------------------------------------------------
def date_shift_frame(split: str) -> pl.DataFrame:
    """[MVT_ID_mvt, days_after_sched] para las filas DEP: fecha(MVT_TIME) − fecha(SCHED_TIME) en días."""
    lf = scan_training(_DATE_RAW) if split == "train" else scan_ranking(_DATE_RAW)
    return (lf.filter(pl.col(PHASE_COL) == DEP)
              .select(ID_COL, (pl.col("MVT_TIME_UTC_mvt").dt.date() - pl.col("SCHED_TIME_UTC_mvt").dt.date())
                      .dt.total_days().cast(pl.Int16).alias(DATE_SHIFT_COL))
              .collect())


def add_date_shift(df: pl.DataFrame, split: str) -> pl.DataFrame:
    out = df.join(date_shift_frame(split), on=ID_COL, how="left")
    assert out.height == df.height and out[DATE_SHIFT_COL].null_count() == 0
    return out


def load_features_v3(split: str, surface: bool) -> pl.DataFrame:
    if surface:
        from v2_features import load_features_v2
        base = load_features_v2(split)
    else:
        from prepare_training import load_features
        base = load_features(split)
    return add_date_shift(base, split)


def shifted_no_aobt3_expr() -> pl.Expr:
    return (pl.col(DATE_SHIFT_COL) >= 1) & (pl.col("has_aobt3").cast(pl.Int64) == 0)


# ---------------------------------------------------------------------------
# Clasificador de régimen, etapa 2
# ---------------------------------------------------------------------------
class RegimeClassifier2:
    """P(clase | x) = (p0, p1, p2). Ver docstring del módulo. `alpha` es el pseudo-recuento de las celdas;
    predict_proba(df, alpha=...) permite evaluar varios α con el mismo ajuste."""
    name = "proba2"

    def __init__(self, num, cat, iterations: int = 300, alpha: float = ALPHA_PRIMARY):
        self.num, self.cat, self.iterations, self.alpha = list(num) + [DATE_SHIFT_COL], list(cat), iterations, float(alpha)
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "RegimeClassifier2":
        from catboost import CatBoostClassifier, Pool
        assert DATE_SHIFT_COL in train.columns
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
        lab_df = train.with_columns(regime_class_expr(), (pl.col(DATE_SHIFT_COL) >= 1).cast(pl.Int8).alias("_shift"))
        glob = float((lab == 2).mean())
        g = lab_df.group_by("ADEP_mvt", "has_aobt3", "_shift").agg(pl.len().alias("n"), (pl.col("regime_class") == 2).sum().alias("k"))
        self.rate2 = {(r["ADEP_mvt"], int(r["has_aobt3"]), int(r["_shift"])): (r["k"] + RATE2_PSEUDO * glob) / (r["n"] + RATE2_PSEUDO)
                      for r in g.iter_rows(named=True)}
        self.rate2_global = glob
        cell_df = lab_df.filter(shifted_no_aobt3_expr()).with_columns((pl.col("mvt_minus_sched").cast(pl.Float64) >= BAND_SEC).cast(pl.Int8).alias("_band"))
        agg = [pl.len().alias("n"), (pl.col("regime_class") == 1).sum().alias("k1"), (pl.col("regime_class") == 2).sum().alias("k2")]
        c = cell_df.group_by("ADEP_mvt").agg(agg)
        self.cells = {r["ADEP_mvt"]: (int(r["n"]), int(r["k1"]), int(r["k2"])) for r in c.iter_rows(named=True)}
        cb = cell_df.group_by("ADEP_mvt", "_band").agg(agg)
        self.cells_band = {(r["ADEP_mvt"], int(r["_band"])): (int(r["n"]), int(r["k1"]), int(r["k2"])) for r in cb.iter_rows(named=True)}
        self.info = {"n_train": int(lab.size), "n_class1": int((lab == 1).sum()), "n_class2": int((lab == 2).sum()),
                     "train_seconds": round(time.time() - t0, 1), "alpha": self.alpha,
                     "cells_shifted_no_aobt3": {k: {"n": v[0], "k1": v[1], "k2": v[2]} for k, v in self.cells.items()},
                     "cells_band": {f"{k[0]}|band{k[1]}": {"n": v[0], "k1": v[1], "k2": v[2]} for k, v in self.cells_band.items()},
                     "feature_importance": dict(zip(feats, np.round(self.clf.get_feature_importance(), 3).tolist()))}
        return self

    def _raw(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self.num + self.cat
        X = self.codec.transform(df.select(feats)).to_pandas()
        p1 = self.clf.predict_proba(X)[:, 1]
        shift = (df[DATE_SHIFT_COL].cast(pl.Int64) >= 1).cast(pl.Int64).to_list()
        keys = list(zip(df["ADEP_mvt"].to_list(), df["has_aobt3"].cast(pl.Int64).to_list(), shift))
        p2 = np.array([self.rate2.get(k, self.rate2_global) for k in keys], dtype=np.float64)
        incell = df.select(shifted_no_aobt3_expr().alias("c"))["c"].fill_null(False).to_numpy()
        return p1, p2, incell

    def predict_proba(self, df: pl.DataFrame, alpha: float | None = None, mode: str = "flat") -> np.ndarray:
        """mode="flat": p_celda = (k + α·prior)/(n + α) con prior = (p1 del clasificador, p2 de la tabla).
        mode="band" (jerárquico, etapa 2c): p_banda = (k_banda + α·p_celda)/(n_banda + α), banda = retraso ≥ 12 h."""
        assert mode in CELL_MODES
        a = self.alpha if alpha is None else float(alpha)
        p1_raw, p2, incell = self._raw(df)
        p1 = np.clip(p1_raw * (1 - p2), 0, 1)
        if a > 0 and incell.any():
            apts = df["ADEP_mvt"].to_numpy()
            band = (df["mvt_minus_sched"].cast(pl.Float64).fill_null(0.0).to_numpy() >= BAND_SEC).astype(int)
            for i in np.where(incell)[0]:
                n, k1, k2 = self.cells.get(apts[i], (0, 0, 0))
                c1 = (k1 + a * p1_raw[i]) / (n + a); c2 = (k2 + a * p2[i]) / (n + a)
                if mode == "band":
                    nb, kb1, kb2 = self.cells_band.get((apts[i], band[i]), (0, 0, 0))
                    c1 = (kb1 + a * c1) / (nb + a); c2 = (kb2 + a * c2) / (nb + a)
                p1[i], p2[i] = c1, c2
        p0 = np.clip(1 - p1 - p2, 0, 1)
        return np.column_stack([p0, p1, p2])


def mixture(p: np.ndarray, normal: np.ndarray, mvt_minus_sched: np.ndarray) -> np.ndarray:
    """E[y|x] = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal). Sin recorte (el recorte a 0 se aplica en la submission)."""
    s = np.nan_to_num(mvt_minus_sched.astype(np.float64), nan=0.0)
    return p[:, 0] * normal + p[:, 1] * s + p[:, 2] * (DAY_SEC + normal)


# ---------------------------------------------------------------------------
# Ejecución de unidades
# ---------------------------------------------------------------------------
def _check_alignment(fold: str, val: pl.DataFrame) -> None:
    meta = np.load(PRED_DIR_V2 / f"meta__{fold}.npz", allow_pickle=False)
    ids = val[ID_COL].to_numpy()
    assert ids.shape == meta["ids"].shape and np.array_equal(ids, meta["ids"]), f"orden/ids de validación distintos a V2 en {fold}"


def run_unit(variant: str, fold: str, feats: pl.DataFrame, resume: bool) -> list[dict]:
    cfg = variants()[variant]
    train, val = split_fold(feats, fold)
    train, _ = apply_outlier_strategy(train, STRATEGY)
    if cfg["surface"]:
        from v2_features import RelativeCongestion
        rc = RelativeCongestion().fit(train)
        train, val = rc.transform(train), rc.transform(val)
    _check_alignment(fold, val)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    y = val[TARGET].to_numpy().astype(np.float64)
    lab = val.select(regime_class_expr())["regime_class"].to_numpy()
    rows = []
    for kind in cfg["kinds"]:
        if kind == "proba":
            files = {("flat", a): PRED_DIR / f"{variant}__{fold}__proba_a{a}.npy" for a in ALPHAS}
            files[("band", ALPHA_PRIMARY)] = PRED_DIR / f"{variant}__{fold}__proba_b{ALPHA_PRIMARY}.npy"
            if resume and all(f.exists() for f in files.values()):
                print(f"  {variant:12s} {fold} proba (cache)", flush=True); continue
            t0 = time.time()
            m = RegimeClassifier2(cfg["num"], cfg["cat"]).fit(train)
            (PRED_DIR / f"{variant}__{fold}__proba__info.json").write_text(json.dumps(m.info, indent=1), encoding="utf-8")
            from sklearn.metrics import log_loss, roc_auc_score
            for (mode, a), f in files.items():
                out = m.predict_proba(val, alpha=a, mode=mode); np.save(f, out.astype(np.float32))
                tag = f"proba_{'a' if mode == 'flat' else 'b'}{a}"
                auc1 = float(roc_auc_score(lab == 1, out[:, 1])) if (lab == 1).any() else np.nan
                auc2 = float(roc_auc_score(lab == 2, out[:, 2])) if (lab == 2).any() else np.nan
                rows.append({"variant": variant, "fold": fold, "model": tag, "n": int(y.size), "fit_seconds": round(time.time() - t0, 1),
                             "logloss": float(log_loss(lab, np.clip(out, 1e-9, 1), labels=[0, 1, 2])), "auc_class1": auc1, "auc_class2": auc2,
                             "mean_p1": float(out[:, 1].mean()), "obs_rate1": float((lab == 1).mean()), "mean_p2": float(out[:, 2].mean()),
                             "obs_rate2": float((lab == 2).mean()), "n_train": m.info["n_train"], "n_class1": m.info["n_class1"], "n_class2": m.info["n_class2"]})
                print(f"  {variant:12s} {fold} {tag:10s} logloss={rows[-1]['logloss']:.4f} auc1={auc1:.3f} auc2={auc2:.3f} "
                      f"p2={rows[-1]['mean_p2']:.5f}/{rows[-1]['obs_rate2']:.5f}", flush=True)
            print(f"  {variant:12s} {fold} proba fit={m.info['train_seconds']}s celdas={m.info['cells_shifted_no_aobt3']} bandas={m.info['cells_band']}", flush=True)
            continue
        f = PRED_DIR / f"{variant}__{fold}__{kind}.npy"
        if resume and f.exists():
            print(f"  {variant:12s} {fold} {kind:15s} (cache)", flush=True); continue
        t0 = time.time()
        tr_k = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        m = make_model(kind, cfg["num"], cfg["cat"]).fit(tr_k)
        pred = m.predict(val); assert np.isfinite(pred).all()
        np.save(f, pred.astype(np.float32))
        fi = m.info.get("feature_importance")
        if fi:
            (PRED_DIR / f"{variant}__{fold}__{kind}__importance.json").write_text(json.dumps(fi), encoding="utf-8")
        rows.append({"variant": variant, "fold": fold, "model": kind, **block(y, pred), "fit_seconds": round(time.time() - t0, 1), "n_train": tr_k.height})
        print(f"  {variant:12s} {fold} {kind:15s} RMSE={rows[-1]['rmse']:7.1f} MAE={rows[-1]['mae']:6.1f} "
              f"RMSE_normal={rows[-1]['rmse_normal']:6.1f} fit={rows[-1]['fit_seconds']}s n_train={tr_k.height:,}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["S2_mixture"], choices=list(variants()))
    ap.add_argument("--folds", nargs="+", default=DEV_FOLDS, choices=list(V2_FOLDS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    if HOLDOUT_FOLD in args.folds:
        print("AVISO: se está abriendo el holdout de diciembre (segunda apertura). Solo con la finalista congelada.", flush=True)
    t0 = time.time()
    feats = load_features_v3("train", surface=any(variants()[v]["surface"] for v in args.variants))
    print(f"features: {feats.shape} ({time.time()-t0:.1f}s)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "units_log.csv"
    rows = pl.read_csv(log).to_dicts() if log.exists() else []
    for fold in args.folds:
        for variant in args.variants:
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("Tanda detenida por --max-seconds; relanzar con --resume"); return
            new = run_unit(variant, fold, feats, args.resume)
            done = {r["model"] for r in new}
            rows = [x for x in rows if not (x["variant"] == variant and x["fold"] == fold and x["model"] in done)] + new
            if rows:
                pl.from_dicts(rows, infer_schema_length=None).write_csv(log)
    print(f"Hecho en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
