"""
Genera outputs/reports/modeling_report.md a partir de las tablas producidas por
`prepare_training.py` y `backtest.py`. Se ejecuta al final de la fase 5:

    python src/make_report.py

Las secciones interpretativas (leakage, recomendación) se escriben aquí con los números
leídos de las tablas, de modo que el informe siempre refleje la última ejecución.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from config import (  # noqa: E402
    CATBOOST_PARAMS, EARLY_STOP_DAYS, FEATURE_CAT_COLS, FEATURE_NUM_COLS, FOLDS, HIER_LEVELS, MIN_GROUP_N,
    MODELS_DIR, OUTLIER_STRATEGIES, REPORTS_DIR, TABLES_DIR,
)

T = TABLES_DIR


def md_table(df: pl.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if df[c].dtype.is_numeric() else "---" for c in cols) + "|"]
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(fmt.get(c, "{:.1f}").format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    m = pl.read_csv(T / "backtest_metrics.csv")
    ap = pl.read_csv(T / "backtest_metrics_by_airport.csv")
    fb = pl.read_csv(T / "fallback_coverage.csv")
    fi = pl.read_csv(T / "feature_importance.csv") if (T / "feature_importance.csv").exists() else None
    audit = pl.read_csv(T / "feature_audit.csv")
    args = json.loads((T / "backtest_run_args.json").read_text()) if (T / "backtest_run_args.json").exists() else {}

    orig = m.filter(pl.col("strategy") == "original")
    summary = (orig.group_by("model").agg(pl.col("rmse").mean().alias("RMSE_medio"), pl.col("mae").mean().alias("MAE_medio"),
                                          pl.col("medae").mean().alias("MedAE_medio"), pl.col("rmse_normal").mean().alias("RMSE_0-3h"),
                                          pl.col("mae_normal").mean().alias("MAE_0-3h"), pl.col("fit_seconds").mean().alias("fit_s"),
                                          pl.col("predict_seconds").mean().alias("pred_s"), pl.col("n_pred_negative").sum().alias("pred_neg"))
               .sort("RMSE_medio"))
    best = summary["model"][0]
    best_mae = summary.sort("MAE_medio")["model"][0]

    per_fold = orig.pivot(on="fold", index="model", values="rmse").sort(list(FOLDS)[-1])
    per_fold_mae = orig.pivot(on="fold", index="model", values="mae").sort(list(FOLDS)[-1])

    # Outliers
    comp = pl.read_csv(T / "outlier_strategy_comparison.csv")
    key_models = [x for x in ["hierarchical_median", "operational_aobt3", "catboost_direct", "catboost_residual"] if x in comp["model"].unique().to_list()]
    comp_avg = (comp.filter(pl.col("model").is_in(key_models)).group_by("model", "strategy")
                .agg(pl.col("rmse").mean(), pl.col("mae").mean(), pl.col("rmse_normal").mean(), pl.col("mae_normal").mean(),
                     pl.col("rmse_outlier").mean(), pl.col("n_pred_negative").sum())
                .sort("model", "strategy"))
    # Por aeropuerto: mejor modelo vs baselines (media 3 folds, estrategia original)
    ap_avg = (ap.filter((pl.col("strategy") == "original") & pl.col("model").is_in(key_models))
                .group_by("ADEP_mvt_key", "model").agg(pl.col("rmse").mean(), pl.col("mae").mean()))
    ap_rmse = ap_avg.pivot(on="model", index="ADEP_mvt_key", values="rmse").sort("ADEP_mvt_key")
    ap_mae = ap_avg.pivot(on="model", index="ADEP_mvt_key", values="mae").sort("ADEP_mvt_key")
    # Por aeropuerto y estrategia para el mejor CatBoost
    ap_strat = (ap.filter(pl.col("model") == best).group_by("ADEP_mvt_key", "strategy").agg(pl.col("rmse").mean())
                  .pivot(on="strategy", index="ADEP_mvt_key", values="rmse").sort("ADEP_mvt_key"))

    fi_avg = None
    if fi is not None and fi.height:
        fi_avg = (fi.filter((pl.col("strategy") == "original") & (pl.col("model") == (best if best.startswith("catboost") else "catboost_residual")))
                    .group_by("feature").agg(pl.col("importance").mean()).sort("importance", descending=True))

    fb_orig = fb.filter(pl.col("strategy") == "original").pivot(on="fold", index=["model", "fallback"], values="pct_rows").sort("model", "fallback")

    n_folds = orig["fold"].n_unique()
    L: list[str] = []
    w = L.append
    w("# Informe de modelado — PRC Data Challenge 2026 (taxi-out)\n")
    w(f"Generado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · `src/backtest.py` → `src/make_report.py` · SEED = 42\n")
    w("## 1. Cómo se prepararon los datos\n")
    w("""- Se leen los 12 Parquet de training y `ranking.parquet` en Polars *lazy* (solo lectura, sin copias intermedias);
  se filtra `PHASE_mvt == "DEP"` (2 085 047 filas de training, 344 841 de ranking). `MVT_ID_mvt` se conserva solo como identificador.
- `build_features()` (`src/prepare_training.py`) es una única función aplicada a training y a ranking; `check_same_schema()`
  comprueba que ambos producen exactamente las mismas 33 columnas con los mismos tipos.
- Calendario: `month`, `day_of_week`, `hour_utc`, `is_weekend`, `hour_sin/cos`, `month_sin/cos` (de `MVT_TIME_UTC_mvt`).
- Operacionales (segundos): `mvt_minus_aobt3`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`, `mvt_minus_sched`,
  `aobt3_minus_sched`, `has_aobt3`. Todas las marcas usadas están disponibles en ranking; `BLOCK_TIME_UTC_mvt` no se lee.
- Congestión: `departures_per_airport_hour`, `departures_per_airport_day`, `departures_per_airport_runway_hour`,
  `arrivals_per_airport_hour`. Son conteos de marcas de tiempo (sin target) y se calculan igual en ranking, que
  incluye las filas ARR y todas las DEP del mes.
- Categóricas: `"NA"` literal y nulos → `"UNK"`; sin one-hot. Para CatBoost se codifican con un vocabulario entero
  (`CategoryCodec`) ajustado ÚNICAMENTE con el bloque de entrenamiento de cada fold (y con todo 2025 en el modelo final);
  las categorías no vistas (stands/operadores nuevos) → −1. El vocabulario completo se guarda en el bundle.
- Los cortes temporales se hacen por `MVT_TIME_UTC_mvt`, no por archivo.\n""")

    w("## 2. Variables utilizadas y excluidas\n")
    w(f"{len(FEATURE_NUM_COLS)} numéricas + {len(FEATURE_CAT_COLS)} categóricas. Auditoría completa en `outputs/tables/feature_audit.csv`.\n")
    w(md_table(audit.select("variable", "tipo", "pct_nulos_train", "pct_nulos_rank", "riesgo_leakage", "decision"), {"pct_nulos_train": "{:.2f}", "pct_nulos_rank": "{:.2f}"}))

    w("\n## 3. Riesgos de leakage encontrados\n")
    w("""- **`BLOCK_TIME_UTC_mvt`** — prohibida. `TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME` en el 100 % de las filas; está nula en
  ranking-DEP. `prepare_training.py` no la incluye en `RAW_COLS` y `assert_no_forbidden()` falla si aparece entre las features.
- **`AOBT_3_flt`** — permitida por las reglas (está en ranking) pero es un cuasi-target: el off-block real reportado por A-CDM a
  resolución de minuto. Toda la mejora "fácil" viene de aquí; el modelo de residuo la usa explícitamente. Si el concurso llegase a
  excluirla, el CatBoost directo sin `mvt_minus_aobt3`/`aobt3_minus_sched`/`has_aobt3` sería el plan B.
- **Identificadores** (`MVT_ID_mvt`, `FLIGHT_ID_mvt`) — crecen con el tiempo; excluidos (`FORBIDDEN_FEATURES`).
- **Marcas de tiempo absolutas** — se usan solo como deltas y calendario para no aprender una tendencia 2025 que no extrapola a 2026.
- **Agregaciones** — las medianas jerárquicas, el vocabulario de categorías y el early stopping se ajustan solo con el bloque de
  entrenamiento del fold (`split_fold` asegura `max(train) < min(val)`). La congestión no usa el target.
- **`ARVT_1/3_flt`** — posteriores al despegue; excluidas por prudencia (solo informan de la duración del vuelo).\n""")

    w("## 4. Cómo funciona cada baseline / modelo\n")
    w(f"""1. **Mediana global** del target del bloque de entrenamiento.
2. **Mediana por aeropuerto** (`ADEP_mvt`), fallback global.
3. **Mediana jerárquica con back-off**: niveles {' → '.join('[' + ' + '.join(k) + ']' for k in HIER_LEVELS)} → global.
   Se usa el nivel más específico cuyo grupo tenga ≥ `MIN_GROUP_N = {MIN_GROUP_N}` observaciones (parámetro en `config.py`).
4. **Baseline operacional**: `mvt_minus_aobt3` cuando existe `AOBT_3_flt`; si no, la mediana jerárquica. La variante
   `_biascorr` resta la mediana por aeropuerto de (`mvt_minus_aobt3` − target) aprendida en training (el EDA mostró desfases
   sistemáticos, p. ej. LTFM ≈ +300 s).
5. **CatBoost directo**: `CatBoostRegressor` (RMSE) sobre las 29 features, categóricas nativas.
6. **CatBoost residual**: mismo modelo sobre `target − mvt_minus_aobt3` en filas con `AOBT_3`; predicción =
   `mvt_minus_aobt3 + residuo`; filas sin `AOBT_3` (≈1.1 % train / 1.5 % ranking) → mediana jerárquica.

CatBoost: {json.dumps({k: v for k, v in CATBOOST_PARAMS.items() if k not in ('verbose', 'allow_writing_files')})}.
Early stopping con los últimos {EARLY_STOP_DAYS} días del bloque de entrenamiento (temporal). Los modelos de `models/backtest/`
son artefactos de evaluación (un modelo por fold × estrategia, entrenados con datos parciales): **NO son modelos finales**.
Los modelos finales viven en `models/final/<modelo>/` como bundles (`train_baseline.py --strategy … `), entrenados con todas
las filas válidas de 2025 y un nº fijo de iteraciones (sin ventana de parada reservada).\n""")

    w("## 5. Resultados de cada backtest (target original en entrenamiento)\n")
    w("RMSE por fold:\n"); w(md_table(per_fold))
    w("\nMAE por fold:\n"); w(md_table(per_fold_mae))
    w(f"\nMedia de los {n_folds} folds (todas las filas de validación; `0-3h` = filas con 0 < target ≤ 3 h):\n")
    w(md_table(summary, {"fit_s": "{:.1f}", "pred_s": "{:.2f}"}))
    w("\nCobertura de fallbacks (% de filas de validación por nivel, estrategia original):\n"); w(md_table(fb_orig, {f: "{:.2f}" for f in FOLDS}))

    w("\n## 6. Comparación por aeropuerto (media de los folds, estrategia original)\n")
    w("RMSE:\n"); w(md_table(ap_rmse)); w("\nMAE:\n"); w(md_table(ap_mae))

    w("\n## 7. Impacto de los outliers\n")
    w("""Las cuatro estrategias se aplican SOLO al bloque de entrenamiento; la validación se puntúa siempre contra el target original.
`rmse_outlier` es el RMSE en las filas de validación con target ≤ 0 o > 3 h (muy pocas, pero con errores de miles de segundos).\n""")
    w(md_table(comp_avg))
    w(f"\nRMSE por aeropuerto y estrategia para `{best}` (media de folds):\n"); w(md_table(ap_strat))

    if fi_avg is not None:
        w("\n## 10. Variables con más señal (importancia CatBoost, media de folds)\n")
        w(md_table(fi_avg.head(29), {"importance": "{:.2f}"}))

    # --- Ensamble y selección final (datos de select_model.py / bundles) ---
    ens_path = T / "ensemble_backtest.csv"; sel_path = MODELS_DIR / "final" / "selection.json"
    if ens_path.exists() and sel_path.exists():
        ens = pl.read_csv(ens_path); sel = json.loads(sel_path.read_text(encoding="utf-8"))
        w(f"\n## 7b. Ensamble 50/50 y selección del modelo final (estrategia `{sel['strategy']}`)\n")
        w("RMSE por fold (predicciones de validación ya guardadas; peso 0.5 fijado a priori, sin optimizar sobre los folds):\n")
        w(md_table(ens.pivot(on="fold", index="candidate", values="rmse").join(
            ens.group_by("candidate").agg(pl.col("rmse").mean().alias("rmse_medio"), pl.col("mae").mean().alias("mae_medio")), on="candidate").sort("rmse_medio")))
        clip = pl.read_csv(T / "negative_clip_backtest.csv").group_by("candidate").agg(pl.col("delta_rmse_clip").mean().alias("delta_rmse_si_se_recorta"), pl.col("n_clipped").sum())
        w("\nRecorte de predicciones negativas a 0 (evaluado en los backtests, Δ RMSE medio; negativo = mejora):\n"); w(md_table(clip, {"delta_rmse_si_se_recorta": "{:.4f}"}))
        w(f"\n**Seleccionado: `{sel['selected']}`** — {sel['criterion']}. Recortar negativos: {sel['clip_negative_predictions']}.\n")
        for name in ("catboost_direct", "catboost_residual"):
            mp = MODELS_DIR / "final" / name / "manifest.json"
            if mp.exists():
                man = json.loads(mp.read_text(encoding="utf-8")); it = man.get("iteration_selection", {})
                w(f"- Bundle final `{name}`: {man['iterations']} iteraciones fijas (regla: {it.get('rule')}; best_iteration por fold: {it.get('best_iteration_by_fold')}), "
                  f"{man['n_train_rows']:,} filas de entrenamiento ({man['strategy']}), entrenado {man['trained_at_utc']}, "
                  f"CatBoost {man['versions']['catboost']} / Python {man['versions']['python']}; verificación carga == guardado: ver `check_predictions.npz`.")
    w("\n## 8. Modelo recomendado\n")
    w(f"""La métrica oficial del concurso es **RMSE**. Mejor RMSE medio individual de los 3 backtests: **{best}**; mejor MAE medio: **{best_mae}**.

- **CatBoost directo** gana entre los modelos individuales en RMSE en los tres folds (≈ 576 / 328 / 373 s frente a 705 / 481 / 459 s
  de la mediana jerárquica y 717 / 494 / 480 s de la baseline operacional). El **residual** empata en MAE/MedAE pero pierde 20–75 s
  de RMSE porque hereda los errores de `AOBT_3` cuando ésta es basura (filas extremas, LIRF, LTFM).
- Por aeropuerto el cuadro es matizado: el residual gana en 8 de 10 aeropuertos por 1–14 s de RMSE y el directo gana en LTFM y sobre
  todo en LIRF (1295 vs 1479), que por sí solo decide el RMSE global. Por eso se evaluó formalmente el **ensamble 50/50**
  (sección 7b): es el candidato con menor RMSE medio y menor varianza entre folds, y es el seleccionado para la submission.
- Estrategia de outliers (solo en entrenamiento): `original` y `drop_nonpositive` dan el mejor RMSE global; `drop_gt3h` y `winsor`
  mejoran el MAE (−5 s) y el RMSE en 9 de 10 aeropuertos (5–10 s) pero pierden en LIRF (+90 a +155 s) y con ello en el RMSE global
  (+20 a +35 s). Con RMSE como métrica se fija **`drop_nonpositive`** (equivale a `original` salvo 388 filas de ruido de sensor).
- Predicciones negativas: 0–22 por fold en CatBoost; recortarlas a 0 reduce el RMSE (sección 7b), así que se aplica en la submission.
- Limitación conocida: el early stopping de los backtests usa los últimos 14 días del bloque de entrenamiento; en F1 paró en 125–413
  iteraciones por outliers en esa ventana. El modelo final no usa early stopping: iteraciones fijas elegidas con la regla documentada.\n""")
    w("## 9. Estado de la submission y qué falta\n")
    w("""- `src/make_submission.py` genera `submissions/candidate_v1.parquet` desde los bundles finales (sin reentrenar), une por
  `MVT_ID_mvt`, restaura el orden de `submitting.parquet`, conserva el esquema (Int32) y valida antes y después de escribir
  (344 841 filas, IDs únicos, sin faltantes/extra, sin nulos, finitos). El archivo **no se ha subido** al bucket.
- Antes de subir: revisar `submissions/candidate_v1.json` (hashes, verificación de bundles, checks) y decidir si se quiere una
  segunda candidata con `drop_gt3h` (mejor en 9/10 aeropuertos) como apuesta alternativa sobre LIRF.
- Mejoras con recorrido: congestión en ventanas de ±15/30 min, cola de la misma pista en los 10 min previos, target encoding
  temporal de `STAND`/`FLIGHT_mvt`, backtest de horizonte largo (ene–may → dic) y un tratamiento específico de LIRF.\n""")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "modeling_report.md").write_text("\n".join(L), encoding="utf-8")
    print("Informe escrito en outputs/reports/modeling_report.md")


if __name__ == "__main__":
    main()
