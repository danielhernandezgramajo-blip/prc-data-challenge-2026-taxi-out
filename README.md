# PRC Data Challenge 2026 — Predicción del taxi-out time

Pipeline reproducible para predecir `TAXITIME_SEC_mvt` (segundos de rodaje desde el
*off-block* hasta el despegue) **solo para movimientos de salida** (`PHASE_mvt == "DEP"`)
y generar una submission compatible con `submitting.parquet`.

Estado: **V1** (fases 1–5, bundles, pruebas, `submissions/candidate_v1.parquet`, congelada) y
**V2 = S_mix** (`submissions/candidate_v2.parquet`, recomendada), evaluada con un protocolo de
origen creciente de 8 folds y holdout de diciembre. **No se ha subido nada al bucket del
concurso.** Métrica oficial: **RMSE**. Licencia: GPL-3.0 (`LICENSE`).

## Estructura

```
Aeropuertos Taxi time/
├── *.parquet                 # 14 Parquet originales (INTACTOS; ver "Datos")
├── data/raw/                 # ubicación alternativa para los Parquet (vacía por ahora)
├── notebooks/01_eda.ipynb    # EDA ejecutable de principio a fin
├── src/
│   ├── config.py             # rutas relativas, columnas, folds, hiperparámetros, SEED=42
│   ├── io_utils.py           # lectura (solo lectura) de los Parquet con Polars
│   ├── validate_data.py      # FASE 1
│   ├── eda.py / run_eda.py   # FASE 2
│   ├── prepare_training.py   # FASE 3: build_features() + feature_audit.csv
│   ├── modeling_utils.py     # métricas, estrategias de outliers, codec de categóricas
│   ├── train_baseline.py     # FASE 4: medianas, baseline operacional, CatBoost directo/residual
│   ├── backtest.py           # FASE 5: 3 backtests temporales × 4 estrategias de outliers (--resume/--repair/--audit)
│   ├── select_model.py       # ensamble 50/50 vs directo con las predicciones de los folds → models/final/selection.json
│   ├── make_submission.py    # bundles finales → submissions/candidate_v1.parquet (validado antes y después de escribir)
│   └── make_report.py        # outputs/reports/modeling_report.md a partir de las tablas
├── tests/                    # pytest (9 pruebas): leakage, esquema, codec, bundle, folds, resume, submission
├── models/                   # excluido de git
│   ├── backtest/             # modelos de EVALUACIÓN (uno por fold × estrategia) — NO son finales
│   ├── backtest_predictions/ # predicciones de validación (.npy) para figuras, ensamble y auditoría
│   └── final/<modelo>/       # bundles finales: model.cbm + codec.json + manifest.json (+ fallback_*.parquet)
├── outputs/{figures,tables,reports}/
├── submissions/              # candidate_v1.parquet + .json de validación (excluido de git; NO subido)
├── requirements.txt / requirements-lock.txt
├── LICENSE                   # GPL-3.0
├── .gitignore
└── README.md
```

## Datos

Los 14 Parquet (12 × `training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet`)
**se dejan donde están** (raíz de la carpeta) y solo se leen. `src/config.py` los busca en
este orden: `$PRC_DATA_DIR` → `data/raw/` → raíz del proyecto. Ningún Parquet, modelo,
credencial ni submission se sube a git (`.gitignore`).

Nota técnica: los archivos fueron exportados desde R; `AIRCRAFT_OPERATOR_flt` lleva el tipo
de extensión `arrow.r.vctrs`. Se carga como `string` fijando
`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` (lo hacen los scripts).

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
```

## Reproducir (todo con rutas relativas; se puede lanzar desde cualquier directorio)

```bash
# FASE 1 — validación (≈5 s) → outputs/reports/data_validation.md
python src/validate_data.py

# FASE 2 — EDA (≈10 s, ~2 GB RAM) → outputs/tables/*.csv, outputs/figures/*.png
python src/run_eda.py
jupyter nbconvert --to html --execute notebooks/01_eda.ipynb --output-dir outputs/reports --output eda_report.html

# FASE 3 — features + auditoría (≈5 s) → outputs/tables/feature_audit.csv
python src/prepare_training.py

# FASE 5 — backtesting temporal (baselines + CatBoost directo y residual, 4 estrategias de outliers)
#   ≈ 40 min con 4 CPU / 3 GB RAM. Escribe resultados de forma incremental; se puede cortar y reanudar:
python src/backtest.py --cb-strategies original drop_nonpositive drop_gt3h winsor_p01_p999 --save-models
python src/backtest.py --cb-strategies original drop_nonpositive drop_gt3h winsor_p01_p999 --save-models --resume   # reanudar
python src/backtest.py --figures-only                                                                                # solo figuras
#   → outputs/tables/backtest_metrics.csv, backtest_metrics_by_airport.csv, outlier_strategy_comparison.csv,
#     fallback_coverage.csv, feature_importance.csv; outputs/figures/model_*.png, error_by_airport.png, …

# Auditoría / reparación de artefactos de backtest (sin reentrenar cuando es posible)
python src/backtest.py --audit --save-models
python src/backtest.py --repair --save-models

# Selección del modelo final (ensamble 50/50 vs directo; peso fijo, sin optimizar sobre los folds)
python src/select_model.py --strategy drop_nonpositive

# Modelos finales: todas las filas válidas de 2025 (drop_nonpositive), iteraciones fijas elegidas con los backtests,
# sin ventana de early stopping; bundle verificado (predicción antes de guardar == tras cargar). ≈ 8 min/modelo (2 CPU).
python src/train_baseline.py --models cb_direct cb_residual --strategy drop_nonpositive

# Submission candidata (NO la sube): carga bundles, features de ranking, ensamble, recorte ≥ 0, unión por MVT_ID_mvt,
# orden y esquema de submitting.parquet, validación antes y después de escribir.
python src/make_submission.py

# Pruebas (≈ 10 s) e informe
python -m pytest tests -q
python src/make_report.py
```

## Bundle reproducible (`models/final/<modelo>/`)

`model.cbm` · `codec.json` (vocabulario completo del `CategoryCodec`, ajustado solo con training) ·
`manifest.json` (orden exacto de features, categóricas, estrategia, iteraciones, seed, versiones de Python/Polars/pandas/NumPy/
CatBoost, fecha, inventario con sha256 de los 14 Parquet, hiperparámetros, importancia) · `check_predictions.npz`
(predicciones de verificación) · `fallback_level*.parquet` (solo residual). `ModelBundle.load()` reconstruye el modelo sin
reentrenar y `ModelBundle.verify()` comprueba que predice exactamente lo mismo que antes de guardarse.

## Diseño del modelado (resumen; detalle en `outputs/reports/modeling_report.md`)

- **Features (29)**: calendario (mes, día de semana, hora, fin de semana, seno/coseno),
  operacionales (`MVT_TIME − AOBT_3/EOBT_1/LOBT/IOBT/SCHED`, `AOBT_3 − SCHED`, `has_aobt3`),
  congestión (salidas por aeropuerto-hora, aeropuerto-día, aeropuerto-pista-hora; llegadas por
  aeropuerto-hora) y 10 categóricas. Una única función `build_features()` para training y ranking.
- **Anti-leakage**: `BLOCK_TIME_UTC_mvt` no se lee; `MVT_ID_mvt` solo trazabilidad; cortes por
  `MVT_TIME_UTC_mvt`; medianas, vocabularios y early stopping solo con el bloque de entrenamiento;
  congestión sin target. Auditoría en `outputs/tables/feature_audit.csv`.
- **Backtests**: ene–jun→jul, ene–oct→nov, ene–nov→dic (2025). Nunca aleatorios.
- **Outliers**: 4 estrategias solo en entrenamiento; validación siempre contra el target original,
  con métricas separadas para todas las filas, 0 < y ≤ 3 h y outliers.

## Hallazgos clave

- Training: 4 167 797 × 30 (2 085 047 DEP); ranking: 689 534 × 30 (344 841 DEP); submitting
  alineado con ranking-DEP. `TAXITIME == MVT_TIME − BLOCK_TIME` exactamente.
- Solo 10 aeropuertos de salida; el ranking contiene únicamente enero y julio de 2026.
- Variables ocultas en ranking-DEP: exactamente `BLOCK_TIME_UTC_mvt` y `TAXITIME_SEC_mvt`;
  `MVT_TIME_UTC_mvt` y `AOBT_3_flt` sí están presentes y son la base de las features operacionales.
- **CatBoost directo** es el mejor modelo individual en los 3 backtests (RMSE medio ≈ 422 s con
  `drop_nonpositive`) frente a la mediana jerárquica (≈ 548) y a la baseline operacional (≈ 563).
  El **ensamble 50/50 directo+residual** (peso fijado a priori) baja el RMSE medio a ≈ 420 s
  (gana en 2 de 3 folds, menor varianza) y es el seleccionado para `candidate_v1`.
- El RMSE global lo dominan 25–85 filas por mes con taxi-out > 3 h (RMSE en esas filas ≈ 20 000 s);
  en el 99.95 % restante el RMSE del CatBoost es de 230–370 s.

## Decisiones documentadas

| Decisión | Motivo |
|---|---|
| Parquet no se mueven a `data/raw/` | Evitar duplicados/pérdida; el código resuelve ambas ubicaciones |
| Polars lazy, sin copia de features en disco | 4.2 M filas en < 2 GB de RAM; `load_features()` tarda 5 s |
| `"NA"` / nulos en categóricas → `"UNK"` | Artefacto de exportación desde R; categoría explícita para CatBoost |
| Codec entero para categóricas ajustado solo con el bloque de entrenamiento | Sin usar el target; categorías nuevas en validación/ranking → −1 |
| `MIN_GROUP_N = 30` en la mediana jerárquica | Medianas de grupos pequeños son ruidosas |
| Early stopping con los últimos 14 días del bloque de entrenamiento (solo backtests) | Temporal; el mes de validación nunca se toca |
| Modelo final sin early stopping: 1000 iteraciones fijas = min(cap, ceil((max best_iteration+1)/100)·100) | F2/F3 llegaron al tope; la parada de F1 fue espuria (outliers en la ventana) |
| Ensamble 50/50 con peso fijo a priori | Evita optimizar pesos sobre los mismos folds que deciden |
| Negativos recortados a 0 en la submission | Δ RMSE ≤ 0 en los tres backtests |
| CatBoost moderado (1000 it., lr 0.1, depth 6, ctr complexity 1) | Cabe en 3 GB y < 3 min por ajuste |
| Outliers no se eliminan en validación | La métrica del concurso se calcula sobre el target real |
| `SEED = 42` en `config.py` | Reproducibilidad |

## V2 — mezcla por régimen (S_mix)

Hallazgo (`src/v2_tail_analysis.py`, `outputs/experiments/v2/tail_*.csv`): cuando el aeropuerto no
registra el off-block real, `BLOCK_TIME_UTC_mvt` se rellena con `SCHED_TIME_UTC_mvt` y el target
vale **exactamente** `MVT_TIME − SCHED_TIME` (100 049 filas de 2025; 170 de los 202 extremos > 3 h).
Una segunda familia (17 filas, LIRF sin plan de vuelo) vale ≈ 86 400 s + taxi. Ambas son funciones
de columnas presentes en ranking, así que V2 predice la esperanza condicional:

    pred = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal),  normal = 0.5·direct_normal + 0.5·residual_V1

`p` = CatBoost binario sin pesos de clase (clase 1) + tasa suavizada por aeropuerto/has_aobt3 (clase 2);
`direct_normal` = CatBoost directo (config V1) entrenado solo con taxi físico (clase 0, ≤ 3 h);
`residual_V1` = bundle V1 sin cambios. Las etiquetas de régimen solo se calculan con el target del
bloque de entrenamiento; validación y ranking nunca las ven.

Protocolo (`src/v2_protocol.py`): folds may→nov (desarrollo) y dic (holdout, abierto una sola vez con
la finalista congelada en `finalist_frozen.json`). Resultados (`outputs/experiments/v2/report.md`):

| | V1 bajo protocolo | V2 S_mix |
|---|---:|---:|
| RMSE pooled desarrollo (may–nov) | 390.6 | **320.9** |
| RMSE holdout diciembre | 332.2 | **241.1** |
| MAE pooled / cobertura ≤ 5 min | 161.1 / 87.71 % | 159.3 / 87.80 % |
| Folds y aeropuertos que mejoran | — | 7/7 · 10/10 |
| Bootstrap pareado por bloques aeropuerto-día, P(mejora) | — | 1.000 (IC 95 % Δ: −94 a −43 s) |

```bash
python src/v2_tail_analysis.py                                             # análisis de regímenes (sin entrenar)
python src/v2_protocol.py --variants v1_protocol S_mixture --resume        # 7 folds de desarrollo
python src/v2_analysis.py                                                  # métricas, bootstrap, conformal, criterios
python src/v2_protocol.py --variants v1_protocol S_mixture --folds Dec --resume && python src/v2_analysis.py --holdout
python src/v2_final.py train && python src/v2_final.py submit             # bundles V2 + candidate_v2.parquet
python src/v2_report.py
```

Implementadas y probadas pero no entrenadas en esta etapa (coste): features de congestión temporal
`src/v2_features.py` (variante `A_surface`). V1 no se modifica: `candidate_v1.parquet` conserva su SHA-256.

## Git y elegibilidad

Repositorio git local inicializado (sin remoto, sin push). `.gitignore` excluye Parquet, `data/`,
`models/`, `submissions/`, `.env*`, credenciales, `*.npy`, `*.cbm`. Comprobado con `git ls-files`.

## Próximos pasos

Ver la sección 9 de `outputs/reports/modeling_report.md`.
