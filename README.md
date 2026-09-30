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
│   ├── make_report.py        # outputs/reports/modeling_report.md a partir de las tablas
│   ├── v2_*.py               # V2: análisis de cola, protocolo 8 folds, features de congestión, mezcla por régimen
│   └── v3_*.py               # V3 (etapa 2): clasificador con celdas de fecha desplazada, A_surface, candidate_v3
├── tests/                    # pytest (20 pruebas): V1 (10), V2 (6), V3 (4)
├── models/                   # excluido de git
│   ├── backtest/             # modelos de EVALUACIÓN (uno por fold × estrategia) — NO son finales
│   ├── backtest_predictions/ # predicciones de validación (.npy) para figuras, ensamble y auditoría
│   └── final/<modelo>/       # bundles finales: model.cbm + codec.json + manifest.json (+ fallback_*.parquet)
├── outputs/{figures,tables,reports}/
├── submissions/              # candidate_v1/v2/v3.parquet + .json; copias para subir optimistic-dragon_v1 (= candidate_v3) y _v2 (= candidate_v2); upload_manifest.json (excluido de git; NO subido)
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

V1 no se modifica: `candidate_v1.parquet` conserva su SHA-256.

## V3 — etapa 2 sobre V2 (S3A_mix)

Motivación (`outputs/experiments/v3/report.md`, sección 1): del error cuadrático restante de S_mix en
desarrollo, el 25 % estaba en **10 filas** de clase 2 (fecha del bloque desplazada un día) y el 16 % en la
clase 1; un clasificador de régimen perfecto bajaría el RMSE pooled de 321 a 231 s. Mecanismo de la clase 2:
en LIRF, sin AOBT_3 y con `fecha(MVT) > fecha(SCHED)`, el off-block se registra con la fecha del programado.

Cambios (mismo protocolo de 8 folds; V2 intacta, sus componentes se reutilizan desde `models/experiments/v2/preds`):

* **2a — clasificador de régimen** (`src/v3_protocol.py`, `RegimeClassifier2`): feature `days_after_sched`
  (fecha MVT − fecha SCHED, ambas en ranking) y, en las celdas «desplazada ≥ 1 día y sin AOBT_3» por aeropuerto,
  actualización bayesiana de p1/p2 con las cuentas del bloque de entrenamiento y pseudo-recuento α = 20 (prefijado;
  0/5/100 solo como ablación/sensibilidad). La ablación «solo feature» da −1 s: la ganancia está en las celdas.
* **2b — rama normal con congestión** (`A_surface`): `direct_normal_s` y `residual_normal_s` con features V1 +
  `src/v2_features.py`, entrenados solo con clase 0 y y ≤ 3 h. Por sí solas −1,8 s (7/7 folds): las features horarias
  de V1 ya capturaban casi toda la congestión; mejoran MAE (−2,3 s) y cobertura ≤ 5 min (+0,36 pp).
* **2c — celdas jerárquicas por banda de retraso (post-hoc, EXPERIMENTAL)**: al abrir diciembre con la finalista
  congelada (S2A_mix) se vio que la celda plana reparte la masa de clase 2 a filas de retraso corto (en 2025: 1/43 con
  retraso < 12 h frente a 10/25 con ≥ 12 h). `proba_b20` añade una segunda actualización por banda con umbral FIJO
  de 12 h elegido tras ver diciembre: S3A_mix se etiqueta como experimental, no como mejora confirmada.
* **2d — umbral elegido dentro de cada ventana de entrenamiento (validación anidada)**: `proba_n20` repite el
  esquema de 2c, pero el umbral (6/9/12/18/24 h, o celda plana) se elige por log-loss leave-one-out **solo con las
  filas de celda del bloque de entrenamiento** de cada fold (May–Aug y Dec eligen 12 h; Sep–Nov, 18 h; todo 2025, 12 h).
  Diciembre nunca interviene en la elección del parámetro; lo que sigue siendo post-hoc es la hipótesis, y eso solo lo
  valida el ranking 2026. S4A_mix es la candidata final (`final_choice.json`); la regla prefijada elige S2A_mix.
* **c8 redefinido** (propuesta de revisión externa, aplicada a posteriori sin cambiar la finalista de la regla): el
  criterio original «la ganancia no depende de pocas filas» es inaplicable a una corrección de régimen raro; se
  sustituye por «no dañar al grueso» (RMSE de la población normal no empeora > 1 s y ≤ 2 de 10 aeropuertos empeoran).

| RMSE (s) | V1P | S_mix (V2) | S2A_mix (regla prefijada) | S3A_mix (experimental) | **S4A_mix (candidate_v3)** |
|---|---:|---:|---:|---:|---:|
| Desarrollo pooled (may–nov) | 390.6 | 320.9 | 300.9 | 297.1 | **297.8** |
| Diciembre (segunda apertura) | 332.2 | 241.1 | 237.8 | 234.4 | **234.4** |
| MAE / cobertura ≤ 5 min (desarrollo) | 161.1 / 87.71 % | 159.3 / 87.80 % | 156.8 / 88.16 % | 156.8 / 88.16 % | **156.8 / 88.16 %** |
| Folds no peores que S_mix · bootstrap P(mejora) | — | — | 7/7 · 1.000 | 7/7 · 1.000 | 7/7 · 1.000 (IC 95 % Δ −36 a −10 s) |

Apuesta frente a parte robusta (frente a S_mix): de los −23,1 s de S4A_mix en desarrollo, solo **−2,4 s** vienen de fuera
de la celda LIRF (features de congestión; mejoran los 10 aeropuertos) y **−20,7 s** de 50 filas de la celda; en diciembre,
−4,1 s fuera y −2,6 s dentro (5 filas). Escalera de riesgo: V1 (control conservador; apuesta implícita, se entrenó con los
mismos targets) < V2 S_mix (apuesta moderada, confirmada) < S2A_mix (ampliación preseleccionada; mejora rigurosamente
defendible) < S3A/S4A (apuesta agresiva y post-hoc en ~25 filas del ranking; S4A es la mejor apuesta). Con todo 2025 el
umbral anidado coincide con 12 h, así que `candidate_v3.parquet` es byte-idéntica a la versión S3A (misma SHA-256).
Todas las desviaciones del protocolo original (E1–E5), la decisión formal sobre c8 (diagnóstico, no veto) y la política de
subida/leaderboard están en `outputs/experiments/v3/protocol_amendments.md`.

Advertencias: la ganancia sigue concentrada en pocas filas (celda LIRF: 68 filas en 2025, **25 en el ranking**);
depende de que el artefacto BLOCK = SCHED persista en 2026 (las llegadas del ranking, donde BLOCK_TIME es visible,
lo muestran con la misma tasa que en 2025); una sola fila de 60 000 s de error mueve el RMSE final ≈ 20 s.

```bash
python src/v3_protocol.py --variants S2_mixture A_surface --resume            # 7 folds de desarrollo (≈ 2,5 h, 2 CPU)
python src/v3_analysis.py --freeze                                            # criterios vs S_mix y V1P; congela la finalista
python src/v3_protocol.py --variants S2_mixture A_surface --folds Dec --resume && python src/v3_analysis.py --holdout
python src/v3_analysis.py --holdout --final-choice S4A_mix --reason "..."     # elección documentada (post-hoc, parámetro anidado)
python src/v3_final.py train && python src/v3_final.py submit                # bundles V3 + candidate_v3.parquet
python src/v3_report.py && python -m pytest tests -q
```

## Git y elegibilidad

Repositorio git local inicializado (sin remoto, sin push). `.gitignore` excluye Parquet, `data/`,
`models/`, `submissions/`, `.env*`, credenciales, `*.npy`, `*.cbm`. Comprobado con `git ls-files`.

## Próximos pasos

Ver la sección 9 de `outputs/reports/modeling_report.md`.
