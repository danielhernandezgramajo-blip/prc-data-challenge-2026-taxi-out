"""Genera outputs/experiments/v3/report.md y feature_audit.csv a partir de las tablas de v3_analysis / v3_final."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS  # noqa: E402
from make_report import md_table  # noqa: E402
from v3_protocol import ALPHA_PRIMARY, DATE_SHIFT_COL, OUT_DIR  # noqa: E402

T = OUT_DIR


def feature_audit() -> pl.DataFrame:
    from v2_features import V2_CAT_COLS, V2_NUM_COLS, V2_REL_COLS
    rows = []
    for c in FEATURE_NUM_COLS + FEATURE_CAT_COLS:
        rows.append({"variable": c, "usada_en": "clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados)",
                     "origen": "features V1 (prepare_training)", "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "usada"})
    rows.append({"variable": DATE_SHIFT_COL, "usada_en": "clasificador etapa 2 (feature) y clave de celdas/tabla de clase 2",
                 "origen": "fecha(MVT_TIME_UTC_mvt) − fecha(SCHED_TIME_UTC_mvt) en días (v3_protocol.date_shift_frame)",
                 "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "usada"})
    rows.append({"variable": "celdas (aeropuerto | desplazada ≥ 1 día, sin AOBT_3): n, k1, k2", "usada_en": "actualización bayesiana de p1/p2 (α = 20)",
                 "origen": "cuentas de clase del bloque de ENTRENAMIENTO", "usa_target": True, "usa_BLOCK_TIME": False, "disponible_en_ranking": False,
                 "estado": "tabla ajustada en entrenamiento (como el codec o las medianas)"})
    for c in V2_NUM_COLS + V2_REL_COLS + V2_CAT_COLS:
        rows.append({"variable": c, "usada_en": "direct_normal_s, residual_normal_s (A_surface)", "origen": "v2_features: solo MVT_TIME/SCHED_TIME/ADEP/ADES/RUNWAY/PHASE",
                     "usa_target": False, "usa_BLOCK_TIME": False, "disponible_en_ranking": True, "estado": "usada (etapa 2b)"})
    df = pl.DataFrame(rows); df.write_csv(T / "feature_audit.csv"); return df


def main() -> None:
    pooled = pl.read_csv(T / "metrics_pooled.csv"); m = pl.read_csv(T / "metrics_by_fold.csv"); boot = pl.read_csv(T / "bootstrap_comparison.csv")
    sel = pl.read_csv(T / "selection_criteria.csv"); ap = pl.read_csv(T / "metrics_by_airport.csv"); reg = pl.read_csv(T / "experiment_registry.csv")
    run = json.loads((T / "analysis_run.json").read_text()); dec = json.loads((T / "decision_dev.json").read_text())
    frozen = json.loads((T / "finalist_frozen.json").read_text()) if (T / "finalist_frozen.json").exists() else None
    units = pl.read_csv(T / "units_log.csv", infer_schema_length=None)
    audit = feature_audit()
    fin = frozen["finalist"] if frozen else dec["finalist"]
    show = ["V1P", "S_mix", "S2_mix_a0", "S2_mix", "A_mix", "A_mix_4", "S2A_mix", "S2A_mix_4", "S3_mix", "S3A_mix", "S3A_mix_4"]
    fc = json.loads((T / "final_choice.json").read_text()) if (T / "final_choice.json").exists() else None
    L = []; w = L.append
    w("# V3 (etapa 2 de V2) — Informe de experimentos (mismo protocolo de origen creciente, 8 folds)\n")
    w(f"Generado {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Holdout de diciembre abierto (segunda apertura): **{run['holdout_opened']}**. "
      f"Finalista congelada antes de abrirlo: **{fin}**" + (f" ({frozen['frozen_at_utc']})" if frozen else "") + ".\n")
    w("## 1. Motivación: dónde estaba el error restante de S_mix (desarrollo, mayo–noviembre)\n")
    w("""Descomposición del error cuadrático de S_mix por clase de régimen: clase 0 (taxi físico) 59 %, clase 1 (bloque = hora
programada) 16 % y **clase 2 (fecha del bloque desplazada un día) 25 % con solo 10 filas**. Un clasificador perfecto bajaría el
RMSE pooled de 321 a 231 s. Mecanismo de la clase 2 (LIRF, sin AOBT_3, fecha de MVT posterior a la programada): el off-block se
registra con la hora correcta pero con la fecha del programado. En 2025 esa celda tiene 68 filas: 54 de clase 1, 11 de clase 2 y
3 normales; el clasificador de V2 les asignaba p2 ≈ 0,005 (predicción 30–60 k s frente a 87 k reales).\n""")
    w("## 2. Qué se probó (registro previo a los resultados) y regla de decisión\n")
    w(md_table(reg))
    w(f"\n{dec['rule']}. α = {ALPHA_PRIMARY} prefijado (0 = ablación sin celdas; 5 y 100 = sensibilidad, no se selecciona entre ellos).\n")
    w("## 3. Cómo se evitó el leakage\n")
    w(f"""- Mismos cortes por `MVT_TIME_UTC_mvt`, misma estrategia (drop_nonpositive) y mismos ids/orden de validación que V2 (verificado).
- `{DATE_SHIFT_COL}` = fecha(MVT) − fecha(SCHED): sin target ni `BLOCK_TIME`; ambas columnas están en ranking.
- Las cuentas de las celdas y la tabla de clase 2 se calculan SOLO con el bloque de entrenamiento de cada fold.
- Features de congestión (etapa 2b) construidas solo con marcas de tiempo/claves; congestión relativa ajustada con el bloque de entrenamiento.
- Validación puntuada contra el target original completo; diciembre sellado hasta congelar la finalista.\n""")
    w(md_table(audit.select("variable", "usada_en", "usa_target", "usa_BLOCK_TIME", "disponible_en_ranking", "estado")))
    w("\n## 4. Unidades ejecutadas (tiempos, AUC del clasificador, RMSE de los componentes)\n")
    cols = [c for c in ["variant", "fold", "model", "fit_seconds", "n_train", "logloss", "auc_class1", "auc_class2", "rmse", "rmse_normal"] if c in units.columns]
    w(md_table(units.select(cols).sort("variant", "fold", "model")))
    w("\n## 5. Resultados pooled de desarrollo (RMSE sobre todas las filas, no media de medias)\n")
    p = pooled.filter((pl.col("set") == "dev_pooled") & pl.col("candidate").is_in(show))
    w(md_table(p.select("candidate", "rmse", "mae", "medae", "cov_le300", "cov_le600", "rmse_class0", "rmse_class1", "rmse_class2", "sse_share_class0", "sse_share_class1", "sse_share_class2", "n_pred_negative"),
               {"cov_le300": "{:.2f}", "cov_le600": "{:.2f}", "sse_share_class0": "{:.1f}", "sse_share_class1": "{:.1f}", "sse_share_class2": "{:.1f}"}))
    w("\nSensibilidad a α (S2_mix): \n")
    w(md_table(pooled.filter((pl.col("set") == "dev_pooled") & pl.col("candidate").is_in(["S_mix", "S2_mix_a0", "S2_mix_a5", "S2_mix", "S2_mix_a100"])).select("candidate", "rmse", "mae", "cov_le300", "rmse_class1", "rmse_class2"), {"cov_le300": "{:.2f}"}))
    w("\n## 6. Por fold (referencias y elegibles)\n")
    w(md_table(m.filter(pl.col("candidate").is_in(show) & (pl.col("set") == "dev")).pivot(on="candidate", index="fold", values="rmse").sort("fold")))
    w("\n## 7. Por aeropuerto (pooled de desarrollo; RMSE)\n")
    a = ap.filter((pl.col("fold") == "dev_pooled") & pl.col("candidate").is_in(["S_mix", fin] if fin else ["S_mix"]))
    w(md_table(a.pivot(on="candidate", index="ADEP_mvt", values="rmse").sort("ADEP_mvt")))
    w("\n## 8. Bootstrap pareado por bloques aeropuerto-día (1 000 remuestreos, SEED 42)\n")
    w(md_table(boot.filter(pl.col("candidate").is_in(show)).select("candidate", "vs", "rmse_baseline", "rmse_candidate", "delta_rmse_point", "delta_p2.5", "delta_p97.5", "p_improve", "n_blocks"), {"p_improve": "{:.3f}"}))
    w("\n## 9. Criterios de selección (solo desarrollo; frente a S_mix y frente a V1P)\n")
    w(md_table(sel.filter(pl.col("candidate").is_in(show)).select("candidate", "vs", "rmse_gain_s", "cov5_gain_pp", "folds_not_worse", "worst_fold_worsening_pct", "min_cov5_fold", "rmse_normal_gain_s", "share_gain_top50_rows", "p_improve_boot", "all_required_criteria", "c8_not_few_rows"),
               {"cov5_gain_pp": "{:.2f}", "share_gain_top50_rows": "{:.3f}", "p_improve_boot": "{:.3f}"}))
    w(f"\nDecisión en desarrollo (regla prefijada): elegibles {dec['eligible']}; cumplen {[c for _, c in dec['passing']]}; finalista **{dec['finalist']}**.\n")
    w("""
### 9b. Etapa 2c (post-hoc, definida DESPUÉS de abrir diciembre)

Al abrir diciembre con la finalista congelada se vio que la celda plana reparte la masa de clase 2 (p2 ≈ 0,13–0,16) también a
filas de retraso corto de la celda, que casi nunca son de clase 2 (en 2025: 1 de 43 con retraso < 12 h frente a 10 de 25 con
retraso ≥ 12 h; test exacto de Fisher p < 0,001), y además el prior del clasificador deja masa en p0 en una banda donde no hay
filas normales. La versión jerárquica (S3*, `proba_b20`) añade dentro de la celda una segunda actualización bayesiana por banda de
retraso (≥ 12 h) con el mismo α = 20. **Esta variante no estaba registrada antes de abrir diciembre**, por lo que diciembre no es
una validación limpia para ella; se evalúa con los mismos criterios en desarrollo y se documenta la elección en `final_choice.json`.
""")
    if fc:
        w(f"\nElección final para `candidate_v3`: **{fc['candidate']}** (post-hoc: {fc['post_hoc']}; finalista congelada: {fc['frozen_finalist']}). Motivo: {fc['reason']}\n")
    if run["holdout_opened"]:
        w("## 10. Holdout de diciembre (segunda apertura; confirmación, no selección)\n")
        w("""Diciembre ya se había abierto en la etapa 1 para comparar S_mix con V1P; por eso aquí NO puede considerarse un holdout
intacto. Se abre una segunda vez, una sola, con la finalista congelada por la regla prefijada, y se reporta como confirmación.\n""")
        w(md_table(pooled.filter(pl.col("set") == "holdout_Dec").select("candidate", "rmse", "mae", "medae", "cov_le300", "cov_le600", "rmse_class0", "rmse_class1", "rmse_class2", "n_pred_negative"), {"cov_le300": "{:.2f}", "cov_le600": "{:.2f}"}))
    d3 = json.loads((T / "decision.json").read_text()) if (T / "decision.json").exists() else None
    if d3:
        w(f"\n## 11. Decisión\n\n**{d3['decision']}** — {d3['reason']}\n")
        if d3.get("candidate_v3"):
            c = d3["candidate_v3"]
            w(f"\n`{c['file']}` · sha256 `{c['sha256']}` · V1 intacta: `{c['candidate_v1_sha256_unchanged']}` · V2 intacta: `{c['candidate_v2_sha256_unchanged']}`\n")
    (T / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("report.md escrito")


if __name__ == "__main__":
    main()
