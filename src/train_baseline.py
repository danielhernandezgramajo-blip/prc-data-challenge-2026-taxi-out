"""
FASE 4 — Baselines y modelos CatBoost.

Todos los modelos siguen la misma interfaz mínima:
    model.fit(train_df)              -> self          (train_df: features + TARGET)
    model.predict(df) -> np.ndarray  (float, una predicción por fila, siempre finita)
    model.info       -> dict         (cobertura de fallbacks, hiperparámetros, tiempos…)

Modelos:
  1. GlobalMedian                   mediana global del target.
  2. AirportMedian                  mediana por ADEP (fallback: global).
  3. HierarchicalMedian             mediana con back-off por niveles (config.HIER_LEVELS) y mínimo
                                    de observaciones por grupo (config.MIN_GROUP_N).
  4. OperationalBaseline            mvt_minus_aobt3 cuando existe; si no, HierarchicalMedian.
  5. CatBoostDirect                 CatBoostRegressor sobre el target.
  6. CatBoostResidual               CatBoostRegressor sobre (target − mvt_minus_aobt3) en filas con AOBT_3;
                                    predicción = mvt_minus_aobt3 + residuo; filas sin AOBT_3 → fallback.

Uso directo (entrena con todo 2025 y guarda modelos en models/):
    python src/train_baseline.py --models hier operational
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import (  # noqa: E402
    CATBOOST_PARAMS, EARLY_STOP_DAYS, FEATURE_CAT_COLS, FEATURE_NUM_COLS, HIER_LEVELS, ID_COL, MIN_GROUP_N,
    MODELS_DIR, OUTLIER_STRATEGIES, SEED, TARGET,
)
from modeling_utils import CategoryCodec, timer  # noqa: E402

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1–3. Medianas
# ---------------------------------------------------------------------------
class GlobalMedian:
    name = "global_median"

    def __init__(self) -> None:
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "GlobalMedian":
        self.median = float(train[TARGET].median())
        self.info = {"median": self.median}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.full(df.height, self.median, dtype=np.float64)


class AirportMedian:
    name = "airport_median"

    def __init__(self, min_n: int = MIN_GROUP_N) -> None:
        self.min_n = min_n
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "AirportMedian":
        self.global_median = float(train[TARGET].median())
        self.table = (train.group_by("ADEP_mvt").agg(pl.col(TARGET).median().alias("_m"), pl.len().alias("_n"))
                      .filter(pl.col("_n") >= self.min_n).select("ADEP_mvt", "_m"))
        self.info = {"n_airports": self.table.height, "global_median": self.global_median}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        p = df.select("ADEP_mvt").join(self.table, on="ADEP_mvt", how="left")["_m"].fill_null(self.global_median)
        return p.to_numpy().astype(np.float64)


class HierarchicalMedian:
    """Mediana por grupo con back-off jerárquico.

    Para cada fila se usa la mediana del nivel más específico cuyo grupo tenga ≥ min_n
    observaciones en el bloque de entrenamiento; si ningún nivel cumple, la mediana global.
    `min_n` está parametrizado (config.MIN_GROUP_N = 30 por defecto): con menos observaciones
    la mediana de un grupo es demasiado ruidosa (p. ej. un stand+pista+hora+tipo raro).
    """
    name = "hierarchical_median"

    def __init__(self, levels: list[list[str]] = HIER_LEVELS, min_n: int = MIN_GROUP_N) -> None:
        self.levels = levels
        self.min_n = min_n
        self.info: dict = {}

    def fit(self, train: pl.DataFrame) -> "HierarchicalMedian":
        self.global_median = float(train[TARGET].median())
        self.tables = []
        for i, keys in enumerate(self.levels):
            t = (train.group_by(keys).agg(pl.col(TARGET).median().alias(f"_m{i}"), pl.len().alias("_n"))
                 .filter(pl.col("_n") >= self.min_n).drop("_n"))
            self.tables.append((keys, t))
        self.info = {"min_n": self.min_n, "levels": [" + ".join(k) for k in self.levels],
                     "groups_per_level": [t.height for _, t in self.tables], "global_median": self.global_median}
        return self

    def predict(self, df: pl.DataFrame, return_level: bool = False):
        d = df.select({c for keys in self.levels for c in keys})
        for i, (keys, t) in enumerate(self.tables):
            d = d.join(t, on=keys, how="left")
        cols = [f"_m{i}" for i in range(len(self.levels))]
        level = pl.coalesce([pl.when(pl.col(c).is_not_null()).then(pl.lit(i + 1)) for i, c in enumerate(cols)] + [pl.lit(len(cols) + 1)])
        pred = pl.coalesce([pl.col(c) for c in cols] + [pl.lit(self.global_median)])
        res = d.select(pred.alias("pred"), level.alias("level"))
        p = res["pred"].to_numpy().astype(np.float64)
        if return_level:
            return p, res["level"].to_numpy()
        return p

    def level_coverage(self, df: pl.DataFrame) -> dict:
        _, lv = self.predict(df, return_level=True)
        names = [" + ".join(k) for k in self.levels] + ["global"]
        return {f"level{i+1}:{names[i]}": round(100 * float((lv == i + 1).mean()), 3) for i in range(len(names))}


# ---------------------------------------------------------------------------
# 4. Baseline operacional
# ---------------------------------------------------------------------------
class OperationalBaseline:
    """pred = MVT_TIME − AOBT_3 cuando AOBT_3 existe; si no, mediana jerárquica.

    Opcionalmente corrige el sesgo mediano de la delta por aeropuerto (el EDA mostró desfases
    sistemáticos, p. ej. LTFM +296 s). `bias_by_airport=True` resta esa mediana aprendida en training.
    """
    name = "operational_aobt3"

    def __init__(self, bias_by_airport: bool = False, fallback: HierarchicalMedian | None = None) -> None:
        self.bias_by_airport = bias_by_airport
        self.fallback = fallback or HierarchicalMedian()
        self.info: dict = {}
        if bias_by_airport:
            self.name = "operational_aobt3_biascorr"

    def fit(self, train: pl.DataFrame) -> "OperationalBaseline":
        self.fallback.fit(train)
        with_a = train.filter(pl.col("mvt_minus_aobt3").is_not_null())
        if self.bias_by_airport:
            self.bias = with_a.group_by("ADEP_mvt").agg((pl.col("mvt_minus_aobt3") - pl.col(TARGET)).median().alias("_b"))
        self.info = {"bias_by_airport": self.bias_by_airport, "pct_train_with_aobt3": round(100 * with_a.height / train.height, 3)}
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        base = df["mvt_minus_aobt3"].cast(pl.Float64)
        if self.bias_by_airport:
            b = df.select("ADEP_mvt").join(self.bias, on="ADEP_mvt", how="left")["_b"].fill_null(0.0)
            base = base - b
        fb = self.fallback.predict(df)
        p = base.to_numpy().astype(np.float64)
        mask = np.isnan(p)
        p[mask] = fb[mask]
        self.info["pct_pred_fallback"] = round(100 * float(mask.mean()), 3)
        return p


# ---------------------------------------------------------------------------
# 5–6. CatBoost
# ---------------------------------------------------------------------------
def _split_early_stop(train: pl.DataFrame, days: int = EARLY_STOP_DAYS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Los últimos `days` días del bloque de entrenamiento se reservan (temporalmente) para early stopping."""
    cutoff = train["MVT_TIME_UTC_mvt"].max() - timedelta(days=days)
    return train.filter(pl.col("MVT_TIME_UTC_mvt") < cutoff), train.filter(pl.col("MVT_TIME_UTC_mvt") >= cutoff)


class _CatBoostBase:
    """Base común. Dos modos de ajuste:
    * backtest (`fixed_iterations=None`): reserva los últimos EARLY_STOP_DAYS días del bloque de
      entrenamiento como conjunto de early stopping (temporal, nunca aleatorio).
    * final (`fixed_iterations=N`): usa TODAS las filas válidas y un nº de iteraciones fijo elegido
      con los backtests (`select_iterations`). Sin conjunto de parada, sin `use_best_model`.
    """

    def __init__(self, params: dict | None = None, codec: CategoryCodec | None = None,
                 fixed_iterations: int | None = None) -> None:
        self.params = {**CATBOOST_PARAMS, **(params or {})}
        self.codec = codec
        self.fixed_iterations = fixed_iterations
        self.features = FEATURE_NUM_COLS + FEATURE_CAT_COLS
        self.cat_idx = [self.features.index(c) for c in FEATURE_CAT_COLS]
        self.info: dict = {}

    def _X(self, df: pl.DataFrame):
        d = self.codec.transform(df.select(self.features))
        return d.to_pandas()

    def _fit_model(self, tr: pl.DataFrame, y_tr: np.ndarray, es: pl.DataFrame | None, y_es: np.ndarray | None):
        from catboost import CatBoostRegressor, Pool
        params = dict(self.params)
        pool_tr = Pool(self._X(tr), y_tr, cat_features=self.cat_idx)
        t0 = time.perf_counter()
        if self.fixed_iterations is not None:
            params["iterations"] = int(self.fixed_iterations)
            params.pop("od_type", None); params.pop("od_wait", None)
            model = CatBoostRegressor(**params)
            model.fit(pool_tr)
            best_it = int(model.tree_count_) - 1
            n_es = 0
        else:
            pool_es = Pool(self._X(es), y_es, cat_features=self.cat_idx)
            model = CatBoostRegressor(**params)
            model.fit(pool_tr, eval_set=pool_es, use_best_model=True)
            best_it = int(model.get_best_iteration())
            n_es = es.height
        self.info.update({"params": {k: v for k, v in params.items() if k != "verbose"},
                          "best_iteration": best_it, "tree_count": int(model.tree_count_),
                          "train_seconds": round(time.perf_counter() - t0, 1),
                          "n_train_rows": tr.height, "n_earlystop_rows": n_es,
                          "feature_importance": dict(zip(self.features, np.round(model.get_feature_importance(), 3).tolist()))})
        return model

    def _fit_split(self, frame: pl.DataFrame):
        if self.fixed_iterations is not None:
            return frame, None
        return _split_early_stop(frame)

    def save(self, path: Path) -> None:
        """Guardado ligero (backtests): .cbm + .json con info. Para el modelo final usar ModelBundle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        (path.with_suffix(".json")).write_text(json.dumps(self.info, indent=2, default=str), encoding="utf-8")


class CatBoostDirect(_CatBoostBase):
    name = "catboost_direct"

    def fit(self, train: pl.DataFrame) -> "CatBoostDirect":
        if self.codec is None:
            self.codec = CategoryCodec(FEATURE_CAT_COLS).fit(train)   # SOLO con el bloque de entrenamiento
        tr, es = self._fit_split(train)
        y_tr = tr[TARGET].to_numpy().astype(np.float32)
        y_es = None if es is None else es[TARGET].to_numpy().astype(np.float32)
        self.model = self._fit_model(tr, y_tr, es, y_es)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self._X(df)).astype(np.float64)


class CatBoostResidual(_CatBoostBase):
    """Aprende r = target − mvt_minus_aobt3 en filas con AOBT_3; predicción = mvt_minus_aobt3 + r̂.
    Filas sin AOBT_3 → `fallback` (por defecto la mediana jerárquica, el mejor fallback validado)."""
    name = "catboost_residual"

    def __init__(self, params: dict | None = None, codec: CategoryCodec | None = None, fallback=None,
                 fixed_iterations: int | None = None) -> None:
        super().__init__(params, codec, fixed_iterations)
        self.fallback = fallback or HierarchicalMedian()

    def fit(self, train: pl.DataFrame) -> "CatBoostResidual":
        if self.codec is None:
            self.codec = CategoryCodec(FEATURE_CAT_COLS).fit(train)
        self.fallback.fit(train)
        with_a = train.filter(pl.col("mvt_minus_aobt3").is_not_null())
        tr, es = self._fit_split(with_a)
        r_tr = (tr[TARGET] - tr["mvt_minus_aobt3"]).to_numpy().astype(np.float32)
        r_es = None if es is None else (es[TARGET] - es["mvt_minus_aobt3"]).to_numpy().astype(np.float32)
        self.model = self._fit_model(tr, r_tr, es, r_es)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        base = df["mvt_minus_aobt3"].cast(pl.Float64).to_numpy().astype(np.float64)
        mask = np.isnan(base)
        p = np.empty(df.height, dtype=np.float64)
        if (~mask).any():
            sub = df.filter(pl.col("mvt_minus_aobt3").is_not_null())
            p[~mask] = base[~mask] + self.model.predict(self._X(sub))
        if mask.any():
            p[mask] = self.fallback.predict(df.filter(pl.col("mvt_minus_aobt3").is_null()))
        self.info["pct_pred_fallback"] = round(100 * float(mask.mean()), 3)
        return p


MODEL_REGISTRY = {
    "global": GlobalMedian, "airport": AirportMedian, "hier": HierarchicalMedian,
    "operational": OperationalBaseline, "operational_bc": lambda: OperationalBaseline(bias_by_airport=True),
    "cb_direct": CatBoostDirect, "cb_residual": CatBoostResidual,
}


# ---------------------------------------------------------------------------
# Selección del nº de iteraciones a partir de los backtests
# ---------------------------------------------------------------------------
def select_iterations(model_name: str, strategy: str, metrics_path: Path | None = None) -> dict:
    """Regla documentada: iteraciones = max(best_iteration de los folds) + 1, redondeado hacia
    arriba a la centena, acotado por CATBOOST_PARAMS['iterations'].

    Motivo: en F2/F3 (1.75–1.9 M filas) el early stopping llegó al tope (≈1000); en F1 la parada
    fue prematura (125–413) por outliers en la ventana de parada, no por sobreajuste. El modelo
    final se entrena con más filas (2.08 M) y sin ventana reservada, por lo que se toma el máximo,
    nunca la media (que estaría contaminada por la parada espuria de F1)."""
    from config import TABLES_DIR
    m = pl.read_csv(metrics_path or TABLES_DIR / "backtest_metrics.csv")
    d = m.filter((pl.col("model") == model_name) & (pl.col("strategy") == strategy))
    if d.height == 0:
        raise ValueError(f"sin backtests para {model_name}/{strategy}")
    best = d.select("fold", "best_iteration").sort("fold")
    cap = CATBOOST_PARAMS["iterations"]
    raw = int(best["best_iteration"].max()) + 1
    chosen = min(cap, int(math.ceil(raw / 100.0) * 100))
    return {"model": model_name, "strategy": strategy, "best_iteration_by_fold": dict(zip(best["fold"].to_list(), best["best_iteration"].to_list())),
            "rule": "min(cap, ceil((max(best_iteration)+1)/100)*100)", "cap": cap, "iterations": chosen}


# ---------------------------------------------------------------------------
# Bundle reproducible
# ---------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def input_inventory(with_hash: bool = True) -> list[dict]:
    """Inventario (nombre, bytes, mtime, sha256) de los 14 Parquet de entrada."""
    from config import DATA_DIR, RANKING_FILE, SUBMITTING_FILE, TRAINING_FILES
    out = []
    for f in [*TRAINING_FILES, RANKING_FILE, SUBMITTING_FILE]:
        p = DATA_DIR / f
        st = p.stat()
        out.append({"file": f, "bytes": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": _sha256(p) if with_hash else None})
    return out


def env_versions() -> dict:
    import catboost, numpy, pandas, polars  # noqa: PLC0415
    return {"python": sys.version.split()[0], "polars": polars.__version__, "pandas": pandas.__version__,
            "numpy": numpy.__version__, "catboost": catboost.__version__, "platform": platform.platform()}


class ModelBundle:
    """Directorio autocontenido para inferencia sin reentrenar:

        <dir>/model.cbm            modelo CatBoost
        <dir>/codec.json           vocabulario completo del CategoryCodec
        <dir>/manifest.json        features (orden exacto), categóricas, estrategia, iteraciones, seed,
                                   versiones, fecha, inventario+sha256 de los 14 Parquet, hiperparámetros
        <dir>/fallback_*.parquet   tablas de la mediana jerárquica (solo catboost_residual)
        <dir>/check_predictions.npz  ids + predicciones de verificación (predicción antes de guardar)
    """

    def __init__(self, model, strategy: str, extra: dict | None = None) -> None:
        self.model = model
        self.strategy = strategy
        self.extra = extra or {}

    def save(self, bundle_dir: Path, check_df: pl.DataFrame | None = None) -> dict:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        m = self.model
        m.model.save_model(str(bundle_dir / "model.cbm"))
        m.codec.save(bundle_dir / "codec.json")
        manifest = {
            "model_name": m.name, "strategy": self.strategy,
            "features": list(m.features), "cat_cols": list(FEATURE_CAT_COLS), "cat_feature_indices": list(m.cat_idx),
            "iterations": int(m.model.tree_count_), "fixed_iterations": m.fixed_iterations,
            "seed": SEED, "catboost_params": m.info.get("params"), "n_train_rows": m.info.get("n_train_rows"),
            "train_seconds": m.info.get("train_seconds"), "feature_importance": m.info.get("feature_importance"),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": env_versions(), "input_files": input_inventory(), **self.extra,
        }
        if isinstance(m, CatBoostResidual):
            fb = m.fallback
            manifest["fallback"] = {"type": "hierarchical_median", "min_n": fb.min_n, "levels": fb.levels,
                                    "global_median": fb.global_median}
            for i, (keys, t) in enumerate(fb.tables):
                t.write_parquet(bundle_dir / f"fallback_level{i}.parquet")
        if check_df is not None:
            pred = m.predict(check_df)
            np.savez_compressed(bundle_dir / "check_predictions.npz", ids=check_df[ID_COL].to_numpy(), pred=pred.astype(np.float64))
            manifest["check"] = {"n_rows": int(check_df.height), "pred_sha256": hashlib.sha256(pred.astype(np.float64).tobytes()).hexdigest()}
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return manifest

    @staticmethod
    def load(bundle_dir: Path):
        """Devuelve (modelo listo para predecir, manifest). No reentrena nada."""
        from catboost import CatBoostRegressor
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        codec = CategoryCodec.load(bundle_dir / "codec.json")
        cb = CatBoostRegressor()
        cb.load_model(str(bundle_dir / "model.cbm"))
        if manifest["model_name"] == "catboost_direct":
            m = CatBoostDirect(codec=codec, fixed_iterations=manifest.get("fixed_iterations"))
        elif manifest["model_name"] == "catboost_residual":
            fbm = manifest["fallback"]
            fb = HierarchicalMedian(levels=fbm["levels"], min_n=fbm["min_n"])
            fb.global_median = fbm["global_median"]
            fb.tables = [(keys, pl.read_parquet(bundle_dir / f"fallback_level{i}.parquet")) for i, keys in enumerate(fbm["levels"])]
            fb.info = {"loaded": True}
            m = CatBoostResidual(codec=codec, fallback=fb, fixed_iterations=manifest.get("fixed_iterations"))
        else:
            raise ValueError(manifest["model_name"])
        assert m.features == manifest["features"], "el orden de features del código no coincide con el del bundle"
        assert list(FEATURE_CAT_COLS) == manifest["cat_cols"], "las categóricas del código no coinciden con las del bundle"
        m.model = cb
        m.info = {"loaded_from": str(bundle_dir), "params": manifest.get("catboost_params")}
        return m, manifest

    @staticmethod
    def verify(bundle_dir: Path, df: pl.DataFrame | None = None) -> dict:
        """Comprueba que la predicción tras cargar == predicción guardada antes de guardar."""
        bundle_dir = Path(bundle_dir)
        m, manifest = ModelBundle.load(bundle_dir)
        chk = np.load(bundle_dir / "check_predictions.npz")
        if df is None:
            from prepare_training import load_features
            df = load_features("rank")
        sub = df.filter(pl.col(ID_COL).is_in(chk["ids"].tolist()))
        # mismo orden que en el guardado
        order = pl.DataFrame({ID_COL: chk["ids"], "_o": np.arange(len(chk["ids"]))})
        sub = sub.join(order, on=ID_COL, how="inner").sort("_o").drop("_o")
        pred = m.predict(sub)
        ok = bool(np.allclose(pred, chk["pred"], rtol=0, atol=1e-6))
        return {"ok": ok, "n": int(sub.height), "max_abs_diff": float(np.max(np.abs(pred - chk["pred"]))),
                "sha256_match": hashlib.sha256(pred.astype(np.float64).tobytes()).hexdigest() == manifest["check"]["pred_sha256"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Entrena modelos con TODO 2025 (filas válidas según --strategy) y guarda bundles en models/final/.

        python src/train_baseline.py --models cb_direct cb_residual --strategy drop_nonpositive

    * La estrategia de outliers se aplica ANTES de entrenar (p. ej. drop_nonpositive elimina target ≤ 0).
    * CatBoost se entrena con un nº fijo de iteraciones elegido con los backtests (select_iterations),
      sin reservar los últimos 14 días.
    * Cada bundle se verifica: predicción antes de guardar == predicción tras cargar.
    La evaluación honesta está en backtest.py; los modelos de models/backtest/ NO son finales."""
    from prepare_training import load_features
    from modeling_utils import apply_outlier_strategy
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cb_direct", "cb_residual"], choices=list(MODEL_REGISTRY))
    ap.add_argument("--strategy", default="drop_nonpositive", choices=OUTLIER_STRATEGIES,
                    help="tratamiento del target en el entrenamiento final")
    ap.add_argument("--iterations", type=int, default=None, help="fuerza el nº de iteraciones (por defecto: select_iterations)")
    ap.add_argument("--out", default=None, help="carpeta de salida (por defecto models/final)")
    ap.add_argument("--check-rows", type=int, default=50_000, help="filas de ranking para la verificación del bundle")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else MODELS_DIR / "final"
    train_all = load_features("train")
    train, sinfo = apply_outlier_strategy(train_all, args.strategy)
    print(f"training 2025: {train_all.height:,} filas DEP → {train.height:,} tras '{args.strategy}'", flush=True)
    rank = load_features("rank")
    check_df = rank.head(args.check_rows)
    for key in args.models:
        t: dict = {}
        if key in ("cb_direct", "cb_residual"):
            name = MODEL_REGISTRY[key]().name
            sel = {"iterations": args.iterations, "rule": "forzado por --iterations"} if args.iterations else select_iterations(name, args.strategy)
            m = MODEL_REGISTRY[key](fixed_iterations=sel["iterations"])
            print(f"{name}: iteraciones fijas = {sel['iterations']} ({sel['rule']})", flush=True)
            with timer(t, "fit"):
                m.fit(train)
            bdir = out_dir / name
            manifest = ModelBundle(m, args.strategy, extra={"iteration_selection": sel, "outlier_info": sinfo,
                                                            "train_period": [str(train["MVT_TIME_UTC_mvt"].min()), str(train["MVT_TIME_UTC_mvt"].max())]}).save(bdir, check_df)
            ver = ModelBundle.verify(bdir, rank)
            assert ver["ok"] and ver["sha256_match"], f"verificación del bundle fallida: {ver}"
            print(f"{name}: entrenado en {t['fit']} s, {manifest['iterations']} árboles → {bdir} | verificación carga: {ver}", flush=True)
        else:
            m = MODEL_REGISTRY[key]()
            with timer(t, "fit"):
                m.fit(train)
            MODELS_DIR.mkdir(exist_ok=True)
            (out_dir / f"{m.name}.json").parent.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{m.name}.json").write_text(json.dumps({**m.info, **t, "strategy": args.strategy}, indent=2, default=str), encoding="utf-8")
            print(f"{m.name}: entrenado en {t['fit']} s → {out_dir}")


if __name__ == "__main__":
    main()
