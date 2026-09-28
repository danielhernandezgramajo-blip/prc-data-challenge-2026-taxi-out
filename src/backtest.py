"""
FASE 5 — Backtesting temporal (nunca aleatorio).

Folds (config.FOLDS), todos sobre 2025:
    F1_jul : train ene–jun  → val jul
    F2_nov : train ene–oct  → val nov
    F3_dec : train ene–nov  → val dic

Garantías anti-leakage:
* Los cortes se hacen por `MVT_TIME_UTC_mvt` (mes), no por archivo.
* Todas las medianas/vocabularios/modelos se ajustan SOLO con el bloque de entrenamiento del fold.
* El early stopping de CatBoost usa los últimos EARLY_STOP_DAYS días del bloque de entrenamiento
  (nunca el mes de validación).
* Las estrategias de outliers se aplican solo al bloque de entrenamiento; la validación se
  puntúa siempre contra el target original.
* Las features de congestión son conteos de marcas de tiempo (sin target): se calculan igual
  en training, validación y ranking.

Uso:
    python src/backtest.py                       # baselines en 3 folds × 4 estrategias + CatBoost (original) en 3 folds
    python src/backtest.py --models global hier  # subconjunto
    python src/backtest.py --cb-strategy-fold F1_jul   # además, CatBoost con las 4 estrategias en ese fold
Salidas: outputs/tables/backtest_*.csv, outlier_strategy_comparison.csv, fallback_coverage.csv,
         feature_importance.csv, outputs/figures/*.png
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import FIGURES_DIR, FOLDS, MODELS_DIR, OUTLIER_STRATEGIES, SEED, TABLES_DIR, TARGET  # noqa: E402
from modeling_utils import apply_outlier_strategy, metric_block, metrics_by_airport, timer  # noqa: E402
from prepare_training import feature_columns, load_features  # noqa: E402
from train_baseline import MODEL_REGISTRY, HierarchicalMedian  # noqa: E402

N_FEATURES = len(feature_columns())

np.random.seed(SEED)
BASELINES = ["global", "airport", "hier", "operational", "operational_bc"]
PRED_DIR = MODELS_DIR / "backtest_predictions"   # predicciones de validación (float32) para las figuras; excluido de git
PRED_DIR.mkdir(parents=True, exist_ok=True)
CATBOOSTS = ["cb_direct", "cb_residual"]
C1, C2, C3, C4, CG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"


def split_fold(feats: pl.DataFrame, fold: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    m0, m1, mv = FOLDS[fold]
    m = pl.col("MVT_TIME_UTC_mvt").dt.month()
    yr = pl.col("MVT_TIME_UTC_mvt").dt.year()
    train = feats.filter((yr == 2025) & (m >= m0) & (m <= m1))
    val = feats.filter((yr == 2025) & (m == mv))
    assert train["MVT_TIME_UTC_mvt"].max() < val["MVT_TIME_UTC_mvt"].min(), "solape temporal train/val"
    return train, val


def run_one(model_key: str, train: pl.DataFrame, val: pl.DataFrame, fold: str, strategy: str, save_dir: Path | None):
    train_s, sinfo = apply_outlier_strategy(train, strategy)
    m = MODEL_REGISTRY[model_key]()
    t: dict = {}
    with timer(t, "fit_seconds"):
        m.fit(train_s)
    with timer(t, "predict_seconds"):
        pred = m.predict(val)
    assert np.isfinite(pred).all(), f"predicciones no finitas en {model_key}/{fold}/{strategy}"
    y = val[TARGET].to_numpy().astype(np.float64)
    row = {"fold": fold, "strategy": strategy, "model": m.name, **metric_block(y, pred), **t,
           "n_train": sinfo["n_train_after"], "n_val": val.height,
           "pct_pred_fallback": m.info.get("pct_pred_fallback"), "best_iteration": m.info.get("best_iteration")}
    by_ap = metrics_by_airport(val, pred).with_columns(pl.lit(fold).alias("fold"), pl.lit(strategy).alias("strategy"), pl.lit(m.name).alias("model"))
    extra = {}
    if isinstance(m, HierarchicalMedian):
        extra["hier_levels"] = m.level_coverage(val)
    if "feature_importance" in m.info:
        extra["feature_importance"] = m.info["feature_importance"]
        if save_dir is not None:
            m.save(save_dir / f"{m.name}__{fold}__{strategy}.cbm")
    print(f"  {m.name:28s} {strategy:16s} RMSE={row['rmse']:7.1f} MAE={row['mae']:6.1f} MedAE={row['medae']:5.1f} "
          f"fit={t['fit_seconds']:6.1f}s", flush=True)
    return row, by_ap, pred, extra


def make_figures(metrics: pl.DataFrame, by_ap: pl.DataFrame, preds: dict, vals: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    orig = metrics.filter(pl.col("strategy") == "original")
    models = orig.group_by("model").agg(pl.col("rmse").mean().alias("r")).sort("r", descending=True)["model"].to_list()
    folds = list(FOLDS)

    # 1. RMSE por modelo y fold (estrategia original)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(folds)
    x = np.arange(len(models))
    for i, (f, c) in enumerate(zip(folds, [C1, C2, C3])):
        d = orig.filter(pl.col("fold") == f)
        vals_ = [d.filter(pl.col("model") == m)["rmse"][0] if d.filter(pl.col("model") == m).height else np.nan for m in models]
        ax.bar(x + (i - (len(folds) - 1) / 2) * w, vals_, w, color=c, label=f)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set(title="RMSE de validación por modelo y backtest (entrenado con el target original)", ylabel="RMSE (s)")
    ax.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=130); plt.close()

    # 2. Error por aeropuerto (media de los 3 folds) para los modelos clave
    key_models = [m for m in ["hierarchical_median", "operational_aobt3", "catboost_direct", "catboost_residual"] if m in models]
    d = (by_ap.filter((pl.col("strategy") == "original") & pl.col("model").is_in(key_models))
              .group_by("ADEP_mvt_key", "model").agg(pl.col("rmse").mean(), pl.col("mae").mean()).sort("ADEP_mvt_key"))
    airports = d["ADEP_mvt_key"].unique().sort().to_list()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met in zip(axes, ["rmse", "mae"]):
        w = 0.8 / len(key_models); x = np.arange(len(airports))
        for i, (m, c) in enumerate(zip(key_models, [C1, C2, C3, C4])):
            v = [d.filter((pl.col("ADEP_mvt_key") == a) & (pl.col("model") == m))[met][0] for a in airports]
            ax.bar(x + (i - (len(key_models) - 1) / 2) * w, v, w, color=c, label=m)
        ax.set_xticks(x); ax.set_xticklabels(airports, rotation=45, ha="right")
        ax.set(title=f"{met.upper()} por aeropuerto (media de los 3 backtests)", ylabel="segundos")
    axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES_DIR / "error_by_airport.png", dpi=130); plt.close()

    # 3–4. Residuales y predicho vs real del mejor modelo (menor RMSE medio) en el último fold
    best = models[-1]
    fold = folds[-1]
    key = (fold, "original", best)
    if key in preds:
        y = vals[fold]; p = preds[key]; r = p - y
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        inside = r[(r > -1500) & (r < 1500)]
        ax[0].hist(inside, bins=150, color=C1)
        ax[0].axvline(0, color=C2, lw=1)
        ax[0].set(title=f"Residuales pred − real · {best} · {fold}\n(|r| < 1500 s: {100*inside.size/r.size:.1f} % de filas)", xlabel="segundos")
        # ECDF real de |error| (sin acumular la cola en un valor): se dibuja hasta el percentil 99.5 y la cola
        # (|error| > ese percentil) se reporta aparte en el título y en outputs/tables/residual_tail.csv.
        ae = np.sort(np.abs(r)); n = ae.size
        ecdf = np.arange(1, n + 1) / n
        p995 = float(np.quantile(ae, 0.995)); p50, p90, p99 = (float(np.quantile(ae, q)) for q in (0.5, 0.9, 0.99))
        show = ae <= p995
        ax[1].step(ae[show], ecdf[show], where="post", color=C1, lw=2)
        for q, v in (("p50", p50), ("p90", p90), ("p99", p99)):
            ax[1].axvline(v, color=CG, lw=0.8, ls="--"); ax[1].text(v, 0.03, f"{q}={v:.0f}s", rotation=90, fontsize=7, va="bottom", ha="right")
        tail_n = int((ae > p995).sum())
        ax[1].set(title=f"ECDF de |error| (eje hasta p99.5 = {p995:.0f} s)\ncola > p99.5: {tail_n} filas ({100*tail_n/n:.2f} %), máx {ae[-1]:.0f} s · detalle en residual_tail.csv",
                  xlabel="|error| (s)", ylabel="fracción de filas ≤ x"); ax[1].set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=130); plt.close()
        tail_q = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
        pl.DataFrame({"model": [best] * len(tail_q), "fold": [fold] * len(tail_q), "quantile": tail_q,
                      "abs_error_s": [float(np.quantile(ae, q)) for q in tail_q],
                      "n_rows_above": [int((ae > np.quantile(ae, q)).sum()) for q in tail_q]}).write_csv(TABLES_DIR / "residual_tail.csv")

        fig, ax = plt.subplots(figsize=(5.5, 5))
        lim = 3600
        mk = (y >= 0) & (y <= lim) & (p >= 0) & (p <= lim)
        ax.hexbin(y[mk], p[mk], gridsize=80, bins="log", cmap="Blues")
        ax.plot([0, lim], [0, lim], color=C2, lw=1)
        ax.set(title=f"Predicho vs real · {best} · {fold} (0–3600 s)", xlabel="taxi-out real (s)", ylabel="predicho (s)")
        plt.tight_layout(); plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=130); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    ap.add_argument("--models", nargs="+", default=BASELINES + CATBOOSTS, choices=list(MODEL_REGISTRY))
    ap.add_argument("--strategies", nargs="+", default=OUTLIER_STRATEGIES, choices=OUTLIER_STRATEGIES,
                    help="estrategias de outliers para las baselines")
    ap.add_argument("--cb-strategies", nargs="+", default=["original"], choices=OUTLIER_STRATEGIES,
                    help="estrategias para CatBoost en todos los folds")
    ap.add_argument("--cb-strategy-fold", default=None, choices=list(FOLDS),
                    help="fold en el que CatBoost se entrena además con TODAS las estrategias")
    ap.add_argument("--cb-iterations", type=int, default=None)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--resume", action="store_true", help="omite las unidades (fold, estrategia, modelo) ya presentes en backtest_metrics.csv")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="no lanza nuevas unidades pasado este tiempo (permite ejecutar en tandas cortas y reanudar con --resume)")
    ap.add_argument("--figures-only", action="store_true", help="solo regenera figuras y tablas derivadas a partir de los CSV existentes (no toca el manifiesto)")
    ap.add_argument("--repair", action="store_true", help="reconstruye artefactos que falten (by_airport, fallback, feature_importance) a partir de .npy/.json y re-ejecuta la unidad si no es posible")
    ap.add_argument("--audit", action="store_true", help="solo lista las unidades incompletas y sale")
    args = ap.parse_args()
    if args.audit:
        a = audit_artifacts(require_model=args.save_models)
        man = audit_manifest(require_model=args.save_models)
        print(json.dumps({"n_units": a["n_units"], "duplicates": a["duplicates"],
                          "incomplete": {"/".join(k): v for k, v in a["incomplete"].items()},
                          "manifest": man}, indent=2, ensure_ascii=False))
        ok = not a["incomplete"] and all(v == 0 for v in a["duplicates"].values()) and man["consistent"]
        print("AUDITORÍA:", "OK" if ok else "INCONSISTENTE")
        raise SystemExit(0 if ok else 1)
    if args.figures_only:
        finalize(args, write_manifest=False); return
    if args.repair:
        repair_units(args); finalize(args, write_manifest=False); return

    t0 = time.time()
    feats = load_features("train")
    print(f"features train DEP: {feats.height:,} × {feats.width}  ({time.time()-t0:.1f}s)", flush=True)
    if args.cb_iterations:
        from config import CATBOOST_PARAMS
        CATBOOST_PARAMS["iterations"] = args.cb_iterations

    rows, by_ap_parts, fallback_rows, fi_rows = [], [], [], []
    done: set = set()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume and (TABLES_DIR / "backtest_metrics.csv").exists():
        prev = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
        by_prev = pl.read_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
        fallback_rows = pl.read_csv(TABLES_DIR / "fallback_coverage.csv").to_dicts() if (TABLES_DIR / "fallback_coverage.csv").exists() else []
        fi_rows = pl.read_csv(TABLES_DIR / "feature_importance.csv").to_dicts() if (TABLES_DIR / "feature_importance.csv").exists() else []
        # Una unidad solo cuenta como hecha si TODOS sus artefactos existen; las incompletas se
        # descartan de las tablas y se recalculan.
        complete, incomplete = [], []
        for u in set(zip(prev["fold"], prev["strategy"], prev["model"])):
            (complete if not missing_artifacts(u, prev, by_prev, fallback_rows, fi_rows, args.save_models) else incomplete).append(u)
        keep = lambda r: (r["fold"], r["strategy"], r["model"]) in set(complete)  # noqa: E731
        rows = [r for r in prev.to_dicts() if keep(r)]
        comp_df = pl.DataFrame({"fold": [u[0] for u in complete], "strategy": [u[1] for u in complete], "model": [u[2] for u in complete]},
                               schema={"fold": pl.String, "strategy": pl.String, "model": pl.String})
        by_ap_parts = [by_prev.join(comp_df, on=["fold", "strategy", "model"], how="semi")] if by_prev.height else []
        fallback_rows = [r for r in fallback_rows if keep(r)]
        fi_rows = [r for r in fi_rows if keep(r)]
        done = set(complete)
        print(f"--resume: {len(done)} unidades completas; {len(incomplete)} incompletas se recalculan: {sorted(incomplete)}", flush=True)
    name_of = {k: (MODEL_REGISTRY[k]().name) for k in MODEL_REGISTRY}
    save_dir = MODELS_DIR / "backtest" if args.save_models else None
    stopped_early = False
    for fold in args.folds:
        train, val = split_fold(feats, fold)
        print(f"\n=== {fold}: train {train.height:,} ({train['MVT_TIME_UTC_mvt'].min():%Y-%m-%d} → {train['MVT_TIME_UTC_mvt'].max():%Y-%m-%d}) "
              f"| val {val.height:,} ({val['MVT_TIME_UTC_mvt'].min():%Y-%m-%d} → {val['MVT_TIME_UTC_mvt'].max():%Y-%m-%d})", flush=True)
        for key in args.models:
            is_cb = key in CATBOOSTS
            strategies = list(args.cb_strategies) if is_cb else list(args.strategies)
            if is_cb and args.cb_strategy_fold == fold:
                strategies = list(dict.fromkeys(strategies + OUTLIER_STRATEGIES))
            for strategy in strategies:
                if (fold, strategy, name_of[key]) in done:
                    continue
                if args.max_seconds and time.time() - t0 > args.max_seconds:
                    stopped_early = True; break
                row, by_ap, pred, extra = run_one(key, train, val, fold, strategy, save_dir)
                rows.append(row); by_ap_parts.append(by_ap); done.add((fold, strategy, row["model"]))
                np.save(PRED_DIR / f"{fold}__{strategy}__{row['model']}.npy", pred.astype(np.float32))
                if "hier_levels" in extra:
                    for lvl, pct in extra["hier_levels"].items():
                        fallback_rows.append({"fold": fold, "strategy": strategy, "model": row["model"], "fallback": lvl, "pct_rows": pct})
                if row.get("pct_pred_fallback") is not None:
                    fallback_rows.append({"fold": fold, "strategy": strategy, "model": row["model"],
                                          "fallback": "sin AOBT_3 → mediana jerárquica", "pct_rows": row["pct_pred_fallback"]})
                if "feature_importance" in extra:
                    for f, v in extra["feature_importance"].items():
                        fi_rows.append({"fold": fold, "strategy": strategy, "model": row["model"], "feature": f, "importance": v})
                # escritura incremental (por si el proceso se interrumpe)
                TABLES_DIR.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(rows).write_csv(TABLES_DIR / "backtest_metrics.csv")
                pl.concat(by_ap_parts, how="vertical_relaxed").write_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
                if fallback_rows:
                    pl.DataFrame(fallback_rows).write_csv(TABLES_DIR / "fallback_coverage.csv")
                if fi_rows:
                    pl.DataFrame(fi_rows).write_csv(TABLES_DIR / "feature_importance.csv")
            if stopped_early:
                break
        if not (PRED_DIR / f"y__{fold}.npy").exists():
            np.save(PRED_DIR / f"y__{fold}.npy", val[TARGET].to_numpy().astype(np.float32))
        del train, val
        if stopped_early:
            break
    if stopped_early:
        print(f"\nTanda detenida por --max-seconds tras {(time.time()-t0)/60:.1f} min; {len(done)} unidades hechas. "
              f"Relanzar con --resume para continuar.")
        return
    finalize(args)
    print(f"\nBacktest completo en {(time.time()-t0)/60:.1f} min. Tablas en outputs/tables/, figuras en outputs/figures/")


# ---------------------------------------------------------------------------
# Integridad de artefactos (usada por --resume, --repair, --audit y los tests)
# ---------------------------------------------------------------------------
NEEDS_FALLBACK = {"hierarchical_median", "operational_aobt3", "operational_aobt3_biascorr", "catboost_residual"}
NEEDS_FI = {"catboost_direct", "catboost_residual"}


def missing_artifacts(unit: tuple, metrics: pl.DataFrame, by_ap: pl.DataFrame, fallback_rows: list, fi_rows: list,
                      require_model: bool = True) -> list[str]:
    """Devuelve la lista de artefactos que faltan para (fold, strategy, model); [] si está completa."""
    fold, strategy, model = unit
    miss = []
    if metrics.filter((pl.col("fold") == fold) & (pl.col("strategy") == strategy) & (pl.col("model") == model)).height != 1:
        miss.append("metrics")
    if not (PRED_DIR / f"{fold}__{strategy}__{model}.npy").exists():
        miss.append("predictions")
    if not (PRED_DIR / f"y__{fold}.npy").exists():
        miss.append("y")
    n_ap = by_ap.filter((pl.col("fold") == fold) & (pl.col("strategy") == strategy) & (pl.col("model") == model)).height if by_ap.height else 0
    if n_ap != 10:
        miss.append(f"by_airport({n_ap}/10)")
    if model in NEEDS_FALLBACK:
        n_fb = sum(1 for r in fallback_rows if (r["fold"], r["strategy"], r["model"]) == unit)
        exp = 5 if model == "hierarchical_median" else 1
        if n_fb != exp:
            miss.append(f"fallback({n_fb}/{exp})")
    if model in NEEDS_FI:
        n_fi = sum(1 for r in fi_rows if (r["fold"], r["strategy"], r["model"]) == unit)
        if n_fi != N_FEATURES:
            miss.append(f"feature_importance({n_fi}/{N_FEATURES})")
        if require_model and not (MODELS_DIR / "backtest" / f"{model}__{fold}__{strategy}.cbm").exists():
            miss.append("model")
    return miss


def audit_artifacts(require_model: bool = True) -> dict:
    """Audita todas las unidades presentes en backtest_metrics.csv. Devuelve {unit: [faltantes]} (solo incompletas)
    y comprueba duplicados en las tablas."""
    metrics = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    by_ap = pl.read_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
    fb = pl.read_csv(TABLES_DIR / "fallback_coverage.csv").to_dicts() if (TABLES_DIR / "fallback_coverage.csv").exists() else []
    fi = pl.read_csv(TABLES_DIR / "feature_importance.csv").to_dicts() if (TABLES_DIR / "feature_importance.csv").exists() else []
    out = {}
    units = sorted(set(zip(metrics["fold"], metrics["strategy"], metrics["model"])))
    for u in units:
        m = missing_artifacts(u, metrics, by_ap, fb, fi, require_model)
        if m:
            out[u] = m
    dup = {
        "metrics": metrics.height - len(units),
        "by_airport": by_ap.height - by_ap.select("fold", "strategy", "model", "ADEP_mvt_key").unique().height,
        "fallback": len(fb) - len({(r["fold"], r["strategy"], r["model"], r["fallback"]) for r in fb}),
        "feature_importance": len(fi) - len({(r["fold"], r["strategy"], r["model"], r["feature"]) for r in fi}),
    }
    return {"incomplete": out, "duplicates": dup, "n_units": len(units)}


def repair_units(args) -> None:
    """Reconstruye los artefactos que falten sin reentrenar cuando es posible:
    by_airport ← .npy + features; fallback ← features (% sin AOBT_3 / niveles jerárquicos);
    feature_importance ← models/backtest/*.json. Si falta metrics/predictions/model, re-ejecuta la unidad."""
    from prepare_training import load_features
    audit = audit_artifacts(require_model=args.save_models)
    if not audit["incomplete"]:
        print("--repair: nada que reparar"); return
    feats = load_features("train")
    metrics = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    by_ap = pl.read_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
    fb_rows = pl.read_csv(TABLES_DIR / "fallback_coverage.csv").to_dicts() if (TABLES_DIR / "fallback_coverage.csv").exists() else []
    fi_rows = pl.read_csv(TABLES_DIR / "feature_importance.csv").to_dicts() if (TABLES_DIR / "feature_importance.csv").exists() else []
    name_to_key = {MODEL_REGISTRY[k]().name: k for k in MODEL_REGISTRY}
    for unit, miss in audit["incomplete"].items():
        fold, strategy, model = unit
        train, val = split_fold(feats, fold)
        print(f"--repair {unit}: faltan {miss}", flush=True)
        if any(x in ("metrics", "predictions", "model", "y") for x in miss):
            row, b, pred, extra = run_one(name_to_key[model], train, val, fold, strategy, MODELS_DIR / "backtest" if args.save_models else None)
            metrics = pl.concat([metrics.filter(~((pl.col("fold") == fold) & (pl.col("strategy") == strategy) & (pl.col("model") == model))), pl.DataFrame([row])], how="vertical_relaxed")
            np.save(PRED_DIR / f"{fold}__{strategy}__{model}.npy", pred.astype(np.float32))
            if not (PRED_DIR / f"y__{fold}.npy").exists():
                np.save(PRED_DIR / f"y__{fold}.npy", val[TARGET].to_numpy().astype(np.float32))
            miss = miss + ["by_airport", "fallback", "feature_importance"]
            fi_src = extra.get("feature_importance"); fb_src = extra.get("hier_levels"); pct_fb = row.get("pct_pred_fallback")
        else:
            pred = np.load(PRED_DIR / f"{fold}__{strategy}__{model}.npy").astype(np.float64)
            fi_src = fb_src = None; pct_fb = None
            jpath = MODELS_DIR / "backtest" / f"{model}__{fold}__{strategy}.json"
            if jpath.exists():
                fi_src = json.loads(jpath.read_text(encoding="utf-8")).get("feature_importance")
            if model == "hierarchical_median":
                fb_src = HierarchicalMedian().fit(apply_outlier_strategy(train, strategy)[0]).level_coverage(val)
            elif model in NEEDS_FALLBACK:
                pct_fb = round(100 * float(val["mvt_minus_aobt3"].is_null().mean()), 3)
        if any(x.startswith("by_airport") for x in miss):
            b = metrics_by_airport(val, pred).with_columns(pl.lit(fold).alias("fold"), pl.lit(strategy).alias("strategy"), pl.lit(model).alias("model"))
            by_ap = pl.concat([by_ap.filter(~((pl.col("fold") == fold) & (pl.col("strategy") == strategy) & (pl.col("model") == model))), b.select(by_ap.columns)], how="vertical_relaxed")
        if any(x.startswith("fallback") for x in miss) and model in NEEDS_FALLBACK:
            fb_rows = [r for r in fb_rows if (r["fold"], r["strategy"], r["model"]) != unit]
            if fb_src:
                fb_rows += [{"fold": fold, "strategy": strategy, "model": model, "fallback": k, "pct_rows": v} for k, v in fb_src.items()]
            if pct_fb is not None:
                fb_rows.append({"fold": fold, "strategy": strategy, "model": model, "fallback": "sin AOBT_3 → mediana jerárquica", "pct_rows": pct_fb})
        if any(x.startswith("feature_importance") for x in miss) and model in NEEDS_FI:
            if fi_src is None:
                raise RuntimeError(f"no hay feature_importance recuperable para {unit}; relanzar con --resume tras borrar la fila de metrics")
            fi_rows = [r for r in fi_rows if (r["fold"], r["strategy"], r["model"]) != unit]
            fi_rows += [{"fold": fold, "strategy": strategy, "model": model, "feature": f, "importance": v} for f, v in fi_src.items()]
        metrics.write_csv(TABLES_DIR / "backtest_metrics.csv")
        by_ap.write_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
        pl.DataFrame(fb_rows).write_csv(TABLES_DIR / "fallback_coverage.csv")
        pl.DataFrame(fi_rows).write_csv(TABLES_DIR / "feature_importance.csv")
    after = audit_artifacts(require_model=args.save_models)
    print("--repair: incompletas tras reparar:", after["incomplete"], "| duplicados:", after["duplicates"])


def audit_manifest(require_model: bool = True) -> dict:
    """Comprueba que backtest_run_args.json (manifiesto de la corrida de entrenamiento) sea coherente con los
    artefactos observados: estrategias presentes, estrategias de CatBoost presentes, modelos guardados cuando
    se exige --save-models, y que no sea un manifiesto de --figures-only (esa opción nunca debe escribirlo)."""
    path = TABLES_DIR / "backtest_run_args.json"
    out = {"exists": path.exists(), "issues": []}
    if not path.exists():
        out["issues"].append("no existe backtest_run_args.json"); out["consistent"] = False; return out
    man = json.loads(path.read_text(encoding="utf-8"))
    metrics = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    name_of = {k: MODEL_REGISTRY[k]().name for k in MODEL_REGISTRY}
    cb_names = {name_of[k] for k in CATBOOSTS}
    obs_strats = sorted(metrics.filter(~pl.col("model").is_in(cb_names))["strategy"].unique().to_list())
    obs_cb_strats = sorted(metrics.filter(pl.col("model").is_in(cb_names))["strategy"].unique().to_list())
    obs_folds = sorted(metrics["fold"].unique().to_list())
    obs_models = sorted({k for k, n in name_of.items() if n in set(metrics["model"].to_list())})
    out["observed"] = {"strategies": obs_strats, "cb_strategies": obs_cb_strats, "folds": obs_folds, "models": obs_models}
    out["manifest"] = {k: man.get(k) for k in ("strategies", "cb_strategies", "cb_strategy_fold", "folds", "models", "save_models", "resume", "figures_only")}
    if man.get("figures_only"):
        out["issues"].append("el manifiesto proviene de --figures-only: no es un manifiesto de corrida de entrenamiento")
    if sorted(man.get("strategies") or []) != obs_strats:
        out["issues"].append(f"estrategias del manifiesto {sorted(man.get('strategies') or [])} != observadas {obs_strats}")
    man_cb = set(man.get("cb_strategies") or [])
    if man.get("cb_strategy_fold"):
        man_cb |= set(OUTLIER_STRATEGIES)
    if sorted(man_cb) != obs_cb_strats:
        out["issues"].append(f"estrategias CatBoost del manifiesto {sorted(man_cb)} != observadas {obs_cb_strats}")
    if sorted(man.get("folds") or []) != obs_folds:
        out["issues"].append(f"folds del manifiesto {sorted(man.get('folds') or [])} != observados {obs_folds}")
    if sorted(man.get("models") or []) != obs_models:
        out["issues"].append(f"modelos del manifiesto {sorted(man.get('models') or [])} != observados {obs_models}")
    cb_units = metrics.filter(pl.col("model").is_in(cb_names)).select("fold", "strategy", "model").unique()
    n_saved = sum((MODELS_DIR / "backtest" / f"{r['model']}__{r['fold']}__{r['strategy']}.cbm").exists() for r in cb_units.iter_rows(named=True))
    out["models_saved"] = f"{n_saved}/{cb_units.height}"
    if require_model and not man.get("save_models"):
        out["issues"].append("se exige --save-models pero el manifiesto dice save_models=false")
    if man.get("save_models") and n_saved != cb_units.height:
        out["issues"].append(f"el manifiesto dice save_models=true pero solo existen {n_saved}/{cb_units.height} modelos")
    out["consistent"] = not out["issues"]
    return out


def finalize(args, write_manifest: bool = True) -> None:
    """Tablas derivadas + figuras a partir de los CSV/NPY ya escritos (idempotente).
    El manifiesto `backtest_run_args.json` solo se escribe en una corrida de entrenamiento."""
    metrics = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    by_ap = pl.read_csv(TABLES_DIR / "backtest_metrics_by_airport.csv")
    comp = metrics.select("model", "strategy", "fold", "n_train", "rmse", "mae", "medae", "rmse_normal", "mae_normal",
                          "rmse_outlier", "mae_outlier", "n_pred_negative")
    comp.write_csv(TABLES_DIR / "outlier_strategy_comparison.csv")
    preds = {}
    for f in PRED_DIR.glob("*__*__*.npy"):
        fold, strategy, model = f.stem.split("__")
        preds[(fold, strategy, model)] = np.load(f).astype(np.float64)
    vals = {fold: np.load(PRED_DIR / f"y__{fold}.npy").astype(np.float64) for fold in FOLDS if (PRED_DIR / f"y__{fold}.npy").exists()}
    make_figures(metrics, by_ap, preds, vals)
    if write_manifest:
        (TABLES_DIR / "backtest_run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
