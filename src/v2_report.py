"""Genera outputs/experiments/v2/report.md y feature_audit.csv a partir de las tablas de v2_analysis / v2_final."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, MODELS_DIR, PROJECT_ROOT  # noqa: E402
from make_report import md_table  # noqa: E402
from v2_protocol import DEV_FOLDS, HOLDOUT_FOLD, OUT_DIR  # noqa: E402

T = OUT_DIR


def feature_audit() -> pl.DataFrame:
    from v2_features import V2_CAT_COLS, V2_NUM_COLS, V2_REL_COLS
    rows = []
    for c in FEATURE_NUM_COLS + FEATURE_CAT_COLS:
        rows.append({"variable": c, "usada_en": "S_mix (clasificador, direct_normal, residual_V1)", "origen": "features V1 (prepare_training)",
                     "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "usada"})
    rows.append({"variable": "regime_class (etiqueta)", "usada_en": "SOLO como etiqueta de entrenamiento del clasificador y filtro de direct_normal",
                 "origen": "derivada del target del bloque de ENTRENAMIENTO: |y − (MVT−SCHED)| ≤ 10 s; y − 86 400 ∈ [0, 3 h]",
                 "usa_target": True, "usa_BLOCK_TIME": False, "disponible_en_ranking": False, "estado": "etiqueta (no feature)"})
    rows.append({"variable": "mvt_minus_sched (en la mezcla)", "usada_en": "valor de la rama clase 1: p1·(MVT − SCHED)", "origen": "MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt",
                 "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "usada"})
    for c in V2_NUM_COLS + V2_REL_COLS + V2_CAT_COLS:
        rows.append({"variable": c, "usada_en": "A_surface (implementada y probada; NO entrenada en esta etapa por coste)", "origen": "v2_features: solo MVT_TIME/SCHED_TIME/ADEP/ADES/RUNWAY/PHASE",
                     "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "no probada"})
    df = pl.DataFrame(rows); df.write_csv(T / "feature_audit.csv"); return df


def main() -> None:
    m = pl.read_csv(T / "metrics_by_fold.csv"); pooled = pl.read_csv(T / "metrics_pooled.csv"); boot = pl.read_csv(T / "bootstrap_comparison.csv")
    sel = pl.read_csv(T / "selection_criteria.csv"); ap = pl.read_csv(T / "metrics_by_airport.csv"); conf = pl.read_csv(T / "conformal_intervals.csv")
    reg = pl.read_csv(T / "experiment_registry.csv"); run = json.loads((T / "analysis_run.json").read_text()); frozen = json.loads((T / "finalist_frozen.json").read_text())
    tail = pl.read_csv(T / "tail_regimes.csv"); coh = pl.read_csv(T / "tail_aobt3_coherence.csv")
    audit = feature_audit()
    L = []; w = L.append
    w("# V2 — Informe de experimentos (protocolo de origen creciente, 8 folds)\n")
    w(f"Generado {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Holdout de diciembre abierto: **{run['holdout_opened']}**. Finalista congelada antes de abrirlo: **{frozen['finalist']}** ({frozen['frozen_at_utc']}).\n")
    w("## 1. Qué se probó y qué no\n")
    w(md_table(reg))
    w("""
No se probó en esta etapa (por coste de cómputo, no por criterio): **A_surface** (features de cola de pista, presión salidas−llegadas,
minutos desde la salida anterior, congestión relativa; implementadas en `v2_features.py` y cubiertas por pruebas), **B** (pesos por
aeropuerto por LOFO; el análisis por régimen `tail_regime_weights_pooled.csv` mostró que la ganancia estaba en el régimen de la cola,
que S_mix resuelve de forma estructural) y **D** (ARIMA/SARIMAX: 12 meses de historia no permiten ajustar estacionalidad anual y la
señal horaria ya está capturada por las features de congestión). Se documentan como pendientes para una etapa 2.\n""")
    w("## 2. El hallazgo que motiva S_mix\n")
    w("""Cuando el aeropuerto no registra el off-block real, `BLOCK_TIME_UTC_mvt` se rellena con `SCHED_TIME_UTC_mvt` y el target vale
**exactamente** `MVT_TIME − SCHED_TIME` (±10 s). En 2025: 100 049 filas (4.8 %), 170 de los 202 extremos > 3 h, 1 506 de las 3 924 filas
entre 1 y 3 h. Una segunda familia (17 filas, LIRF sin plan de vuelo) vale ≈ 86 400 s + taxi (fecha del bloque desplazada un día).
Ninguno de los 202 extremos coincide con `MVT − AOBT_3`. Ambas familias son funciones conocidas de columnas presentes en ranking, así que
la predicción RMSE-óptima es la esperanza condicional: `p0·normal + p1·(MVT−SCHED) + p2·(86 400 + normal)`.\n""")
    w(md_table(coh)); w("\n"); w(md_table(tail.select("regime", "n", "pct_rows", "p_extreme", "mean_y", "median_y", "pct_of_SSE_vs_global_median"), {"pct_rows": "{:.3f}", "p_extreme": "{:.4f}", "mean_y": "{:.0f}", "median_y": "{:.0f}"}))
    w("\n## 3. Cómo se evitó el leakage\n")
    w("""- Cortes por `MVT_TIME_UTC_mvt`; cada fold entrena solo con meses anteriores al de validación; diciembre sellado hasta congelar la finalista.
- Las etiquetas de régimen se calculan **solo** con el target del bloque de entrenamiento; en validación y ranking nunca se leen.
- El clasificador y `direct_normal` usan exactamente las 29 features V1 (sin target, sin `BLOCK_TIME`); el residual V1 no se reentrena.
- `mvt_minus_sched` (rama clase 1) es `MVT_TIME − SCHED_TIME`, ambas presentes en ranking. Auditoría en `feature_audit.csv`.
- La validación se puntúa contra el target original completo (extremos incluidos); ninguna fila se excluye de la validación.\n""")
    w(md_table(audit.select("variable", "usada_en", "usa_target", "usa_BLOCK_TIME", "disponible_en_ranking", "estado")))
    w("\n## 4. Resultados pooled de desarrollo (mayo–noviembre; RMSE calculado sobre todas las filas, no media de medias)\n")
    w(md_table(pooled.filter(pl.col("set") == "dev_pooled").select("candidate", "rmse", "rmse_unweighted_fold_mean", "mae", "medae", "cov_le300", "cov_le600", "cov_le900", "ae_p87", "ae_p90", "ae_p95", "ae_p99", "ae_max", "rmse_normal", "rmse_extreme", "n_pred_negative", "pct_pred_negative"), {"cov_le300": "{:.2f}", "cov_le600": "{:.2f}", "cov_le900": "{:.2f}", "pct_pred_negative": "{:.3f}"}))
    w("\n## 5. Por fold (V1P vs S_mix)\n")
    w(md_table(m.filter(pl.col("candidate").is_in(["V1P", "S_mix"]) & (pl.col("set") == "dev")).select("candidate", "fold", "n", "rmse", "mae", "medae", "cov_le300", "cov_le600", "cov_le900", "n_extreme", "rmse_extreme", "rmse_normal", "n_pred_negative"), {"cov_le300": "{:.2f}", "cov_le600": "{:.2f}", "cov_le900": "{:.2f}"}))
    w("\n## 6. Por aeropuerto (pooled de desarrollo)\n")
    a = ap.filter((pl.col("fold") == "dev_pooled") & pl.col("candidate").is_in(["V1P", "S_mix"]))
    w(md_table(a.pivot(on="candidate", index="ADEP_mvt", values="rmse").with_columns((pl.col("S_mix") - pl.col("V1P")).alias("delta_rmse")).sort("ADEP_mvt")))
    w("\n"); w(md_table(a.pivot(on="candidate", index="ADEP_mvt", values="mae").with_columns((pl.col("S_mix") - pl.col("V1P")).alias("delta_mae")).sort("ADEP_mvt")))
    w("\n## 7. Bootstrap pareado por bloques aeropuerto-día (1 000 remuestreos, SEED 42, vs V1P)\n")
    w(md_table(boot.select("candidate", "rmse_baseline", "rmse_candidate", "delta_rmse_point", "delta_p2.5", "delta_p50", "delta_p97.5", "p_improve", "n_blocks"), {"p_improve": "{:.3f}"}))
    w("\n## 8. Criterios de selección (solo desarrollo)\n")
    w(md_table(sel.select("candidate", "rmse_gain_s", "cov5_gain_pp", "folds_not_worse", "worst_fold_worsening_pct", "min_cov5_fold", "rmse_normal_gain_s", "share_gain_top50_rows", "p_improve_boot", "all_criteria"), {"cov5_gain_pp": "{:.2f}", "share_gain_top50_rows": "{:.3f}", "p_improve_boot": "{:.3f}"}))
    w("""
Lectura del criterio 8 ("no depender de pocas filas"): el 94 % de la reducción de SSE de S_mix procede de las 50 filas con mayor
ganancia, **por construcción**: la mejora estructural está en el régimen de bloque = hora programada, que por definición son pocas filas
con errores de miles de segundos. No es un artefacto de un fold ni de un aeropuerto: S_mix mejora el RMSE en 7/7 folds (de −7.6 a −143 s),
en los 10 aeropuertos, el MAE en los 10 aeropuertos, el RMSE de la población normal (0 < y ≤ 3 h) en 2.9 s y reduce las predicciones
negativas (103 → 69). Se acepta el candidato con esa salvedad explícita.\n""")
    w("## 9. Intervalos conformales (diagnóstico; calibrados con los folds anteriores, evaluados en el siguiente)\n")
    w(md_table(conf.group_by("candidate", "nominal").agg(pl.col("coverage").mean().alias("cobertura_media"), pl.col("coverage").min().alias("cobertura_min"), pl.col("q_half_width_s").mean().alias("semiancho_medio_s")).sort("candidate", "nominal"), {"cobertura_media": "{:.4f}", "cobertura_min": "{:.4f}", "semiancho_medio_s": "{:.0f}"}))
    w("\nNo se incluyen intervalos en la submission (el concurso pide predicción puntual).\n")
    if run["holdout_opened"]:
        w("## 10. Holdout de diciembre (abierto una sola vez, con la finalista ya congelada)\n")
        w(md_table(pooled.filter(pl.col("set") == "holdout_Dec").select("candidate", "rmse", "mae", "medae", "cov_le300", "cov_le600", "cov_le900", "n_extreme", "rmse_extreme", "rmse_normal", "n_pred_negative"), {"cov_le300": "{:.2f}", "cov_le600": "{:.2f}", "cov_le900": "{:.2f}"}))
    dec = json.loads((T / "decision.json").read_text()) if (T / "decision.json").exists() else None
    if dec:
        w(f"\n## 11. Decisión\n\n**{dec['decision']}** — {dec['reason']}\n")
        if dec.get("candidate_v2"):
            w(f"\n`{dec['candidate_v2']['file']}` · sha256 `{dec['candidate_v2']['sha256']}` · V1 intacta: `{dec['candidate_v2']['candidate_v1_sha256_unchanged']}`\n")
    (T / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("report.md escrito")


if __name__ == "__main__":
    main()
