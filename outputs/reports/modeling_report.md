# Informe de modelado — PRC Data Challenge 2026 (taxi-out)

Generado: 2026-09-28 02:27 UTC · `src/backtest.py` → `src/make_report.py` · SEED = 42

## 1. Cómo se prepararon los datos

- Se leen los 12 Parquet de training y `ranking.parquet` en Polars *lazy* (solo lectura, sin copias intermedias);
  se filtra `PHASE_mvt == "DEP"` (2 085 047 filas de training, 344 841 de ranking). `MVT_ID_mvt` se conserva solo como identificador.
- `build_features()` (`src/prepare_training.py`) es una única función aplicada a training y a ranking; `check_same_schema()`
  comprueba que ambos producen exactamente las mismas 33 columnas con los mismos tipos.
- Calendario: `month`, `day_of_week`, `hour_utc`, `is_weekend`, `hour_sin/cos`, `month_sin/cos` (de `MVT_TIME_UTC_mvt`).
- Operacionales (segundos): `mvt_minus_aobt3`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`, `mvt_minus_sched`,
  `aobt3_minus_sched`, `has_aobt3`. Todas las marcas usadas están disponibles en ranking; `BLOCK_TIME_UTC_mvt` no se lee.
- Congestión: `departures_per_airport_hour`, `departures_per_airport_day`, `departures_per_airport_runway_hour`,
  `arrivals_per_airport_hour`. Son conteos de marcas de tiempo (sin target) y se calculan igual en ranking, que
  incluye las filas ARR y todas las DEP del mes.
- Categóricas: `"NA"` literal y nulos → `"UNK"`; sin one-hot. Para CatBoost se codifican con un vocabulario entero estable
  (training ∪ ranking, sin target); categorías nuevas → −1.
- Los cortes temporales se hacen por `MVT_TIME_UTC_mvt`, no por archivo.

## 2. Variables utilizadas y excluidas

19 numéricas + 10 categóricas. Auditoría completa en `outputs/tables/feature_audit.csv`.

| variable | tipo | pct_nulos_train | pct_nulos_rank | riesgo_leakage | decision |
|---|---|---:|---:|---|---|
| month | calendario | 0.00 | 0.00 | NULO | incluir |
| day_of_week | calendario | 0.00 | 0.00 | NULO | incluir |
| hour_utc | calendario | 0.00 | 0.00 | NULO | incluir |
| is_weekend | calendario | 0.00 | 0.00 | NULO | incluir |
| hour_sin | calendario | 0.00 | 0.00 | NULO | incluir |
| hour_cos | calendario | 0.00 | 0.00 | NULO | incluir |
| month_sin | calendario | 0.00 | 0.00 | NULO | incluir |
| month_cos | calendario | 0.00 | 0.00 | NULO | incluir |
| mvt_minus_aobt3 | operacional | 1.08 | 1.53 | ALTO (permitido) | incluir |
| mvt_minus_eobt1 | operacional | 1.08 | 1.53 | BAJO | incluir |
| mvt_minus_lobt | operacional | 1.08 | 1.53 | BAJO | incluir |
| mvt_minus_iobt | operacional | 1.08 | 1.53 | BAJO | incluir |
| mvt_minus_sched | operacional | 0.00 | 0.00 | BAJO | incluir |
| aobt3_minus_sched | operacional | 1.08 | 1.53 | MEDIO | incluir |
| has_aobt3 | operacional | 0.00 | 0.00 | BAJO | incluir |
| departures_per_airport_hour | congestión | 0.00 | 0.00 | BAJO | incluir |
| departures_per_airport_day | congestión | 0.00 | 0.00 | BAJO | incluir |
| departures_per_airport_runway_hour | congestión | 0.00 | 0.00 | BAJO | incluir |
| arrivals_per_airport_hour | congestión | 0.00 | 0.00 | BAJO | incluir |
| ADEP_mvt | categórica | 0.00 | 0.00 | BAJO | incluir |
| ADES_mvt | categórica | 0.01 | 0.00 | BAJO | incluir |
| RUNWAY_mvt | categórica | 0.00 | 0.00 | BAJO | incluir |
| STAND_mvt | categórica | 0.00 | 0.00 | BAJO | incluir |
| AIRCRAFT_TYPE_mvt | categórica | 0.07 | 0.11 | BAJO | incluir |
| MARKET_SEGMENT_flt | categórica | 1.08 | 1.53 | BAJO | incluir |
| FLIGHT_TYPE_flt | categórica | 1.08 | 1.53 | BAJO | incluir |
| WK_TBL_CAT_flt | categórica | 1.08 | 1.54 | BAJO | incluir |
| AIRCRAFT_OPERATOR_flt | categórica | 1.08 | 1.53 | BAJO | incluir |
| FLIGHT_RULE_mvt | categórica | 0.01 | 0.01 | BAJO | incluir |
| TAXITIME_SEC_mvt | numérica (target) |  |  | TARGET | excluir |
| BLOCK_TIME_UTC_mvt | temporal |  |  | PROHIBIDO | excluir |
| MVT_ID_mvt | identificador |  |  | MEDIO (correlaciona con el tiempo) | excluir (solo trazabilidad) |
| FLIGHT_ID_mvt | identificador |  |  | MEDIO | excluir |
| FLIGHT_mvt / CALLSIGN_flt | categórica |  |  | BAJO | excluir (por ahora) |
| ADEP_flt / AIRCRAFT_TYPE_flt / FLIGHT_RULE_flt / ADES_FILED_flt | categórica |  |  | BAJO | excluir |
| ADES_flt | categórica |  |  | BAJO | excluir |
| ARVT_1_flt / ARVT_3_flt | temporal |  |  | BAJO | excluir |
| PHASE_mvt | categórica |  |  | NULO | excluir |
| MVT_TIME_UTC_mvt y demás marcas absolutas | temporal |  |  | BAJO | excluir como valor absoluto |

## 3. Riesgos de leakage encontrados

- **`BLOCK_TIME_UTC_mvt`** — prohibida. `TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME` en el 100 % de las filas; está nula en
  ranking-DEP. `prepare_training.py` no la incluye en `RAW_COLS` y `assert_no_forbidden()` falla si aparece entre las features.
- **`AOBT_3_flt`** — permitida por las reglas (está en ranking) pero es un cuasi-target: el off-block real reportado por A-CDM a
  resolución de minuto. Toda la mejora "fácil" viene de aquí; el modelo de residuo la usa explícitamente. Si el concurso llegase a
  excluirla, el CatBoost directo sin `mvt_minus_aobt3`/`aobt3_minus_sched`/`has_aobt3` sería el plan B.
- **Identificadores** (`MVT_ID_mvt`, `FLIGHT_ID_mvt`) — crecen con el tiempo; excluidos (`FORBIDDEN_FEATURES`).
- **Marcas de tiempo absolutas** — se usan solo como deltas y calendario para no aprender una tendencia 2025 que no extrapola a 2026.
- **Agregaciones** — las medianas jerárquicas, el vocabulario de categorías y el early stopping se ajustan solo con el bloque de
  entrenamiento del fold (`split_fold` asegura `max(train) < min(val)`). La congestión no usa el target.
- **`ARVT_1/3_flt`** — posteriores al despegue; excluidas por prudencia (solo informan de la duración del vuelo).

## 4. Cómo funciona cada baseline / modelo

1. **Mediana global** del target del bloque de entrenamiento.
2. **Mediana por aeropuerto** (`ADEP_mvt`), fallback global.
3. **Mediana jerárquica con back-off**: niveles [ADEP_mvt + RUNWAY_mvt + hour_utc + AIRCRAFT_TYPE_mvt] → [ADEP_mvt + RUNWAY_mvt + hour_utc] → [ADEP_mvt + RUNWAY_mvt] → [ADEP_mvt] → global.
   Se usa el nivel más específico cuyo grupo tenga ≥ `MIN_GROUP_N = 30` observaciones (parámetro en `config.py`).
4. **Baseline operacional**: `mvt_minus_aobt3` cuando existe `AOBT_3_flt`; si no, la mediana jerárquica. La variante
   `_biascorr` resta la mediana por aeropuerto de (`mvt_minus_aobt3` − target) aprendida en training (el EDA mostró desfases
   sistemáticos, p. ej. LTFM ≈ +300 s).
5. **CatBoost directo**: `CatBoostRegressor` (RMSE) sobre las 29 features, categóricas nativas.
6. **CatBoost residual**: mismo modelo sobre `target − mvt_minus_aobt3` en filas con `AOBT_3`; predicción =
   `mvt_minus_aobt3 + residuo`; filas sin `AOBT_3` (≈1.1 % train / 1.5 % ranking) → mediana jerárquica.

CatBoost: {"loss_function": "RMSE", "iterations": 1000, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3, "border_count": 128, "max_ctr_complexity": 1, "one_hot_max_size": 12, "random_seed": 42, "thread_count": 4, "od_type": "Iter", "od_wait": 80}.
Early stopping con los últimos 14 días del bloque de entrenamiento (temporal). Modelos guardados en `models/backtest/`.

## 5. Resultados de cada backtest (target original en entrenamiento)

RMSE por fold:

| model | F1_jul | F2_nov | F3_dec |
|---|---:|---:|---:|
| catboost_direct | 575.8 | 328.1 | 372.7 |
| catboost_residual | 618.5 | 404.4 | 377.0 |
| hierarchical_median | 704.8 | 480.8 | 459.0 |
| operational_aobt3_biascorr | 710.9 | 480.4 | 468.0 |
| operational_aobt3 | 716.6 | 493.8 | 480.1 |
| airport_median | 725.0 | 507.3 | 490.4 |
| global_median | 754.4 | 538.9 | 521.4 |

MAE por fold:

| model | F1_jul | F2_nov | F3_dec |
|---|---:|---:|---:|
| catboost_residual | 185.6 | 152.4 | 154.9 |
| catboost_direct | 191.9 | 153.0 | 158.0 |
| operational_aobt3_biascorr | 259.8 | 220.6 | 221.2 |
| hierarchical_median | 247.2 | 225.9 | 227.4 |
| operational_aobt3 | 272.4 | 238.7 | 238.7 |
| airport_median | 284.7 | 260.9 | 262.4 |
| global_median | 322.7 | 298.6 | 295.2 |

Media de los 3 folds (todas las filas de validación; `0-3h` = filas con 0 < target ≤ 3 h):

| model | RMSE_medio | MAE_medio | MedAE_medio | RMSE_0-3h | MAE_0-3h | fit_s | pred_s | pred_neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost_direct | 425.5 | 167.6 | 114.8 | 292.8 | 165.4 | 91.1 | 0.26 | 48 |
| catboost_residual | 466.6 | 164.3 | 111.9 | 271.9 | 161.1 | 190.9 | 0.33 | 108 |
| hierarchical_median | 548.2 | 233.5 | 157.5 | 380.2 | 230.1 | 0.1 | 0.01 | 38 |
| operational_aobt3_biascorr | 553.1 | 233.9 | 151.3 | 401.4 | 230.7 | 0.1 | 0.01 | 550 |
| operational_aobt3 | 563.5 | 249.9 | 175.7 | 415.3 | 246.7 | 0.1 | 0.01 | 357 |
| airport_median | 574.3 | 269.3 | 189.7 | 411.4 | 265.8 | 0.0 | 0.00 | 0 |
| global_median | 604.9 | 305.5 | 227.0 | 451.9 | 302.0 | 0.0 | 0.00 | 0 |

Cobertura de fallbacks (% de filas de validación por nivel, estrategia original):

| model | fallback | F1_jul | F2_nov | F3_dec |
|---|---|---:|---:|---:|
| catboost_residual | sin AOBT_3 → mediana jerárquica |  | 0.83 | 0.99 |
| hierarchical_median | level1:ADEP_mvt + RUNWAY_mvt + hour_utc + AIRCRAFT_TYPE_mvt | 85.92 | 88.98 | 91.40 |
| hierarchical_median | level2:ADEP_mvt + RUNWAY_mvt + hour_utc | 11.88 | 10.80 | 8.48 |
| hierarchical_median | level3:ADEP_mvt + RUNWAY_mvt | 2.18 | 0.22 | 0.10 |
| hierarchical_median | level4:ADEP_mvt | 0.02 | 0.00 | 0.02 |
| hierarchical_median | level5:global | 0.00 | 0.00 | 0.00 |
| operational_aobt3 | sin AOBT_3 → mediana jerárquica | 2.06 | 0.83 | 0.99 |
| operational_aobt3_biascorr | sin AOBT_3 → mediana jerárquica | 2.06 | 0.83 | 0.99 |

## 6. Comparación por aeropuerto (media de los folds, estrategia original)

RMSE:

| ADEP_mvt_key | catboost_residual | operational_aobt3 | catboost_direct | hierarchical_median |
|---|---:|---:|---:|---:|
| EDDF | 206.9 | 275.4 | 204.5 | 312.1 |
| EDDM | 215.6 | 421.0 | 219.5 | 355.1 |
| EGLL | 273.9 | 445.5 | 276.1 | 431.6 |
| EHAM | 182.3 | 321.6 | 184.6 | 236.7 |
| LEBL | 222.9 | 323.1 | 224.1 | 311.4 |
| LEMD | 193.9 | 271.9 | 195.8 | 281.8 |
| LFPG | 304.5 | 426.4 | 305.2 | 429.7 |
| LIRF | 1478.6 | 1529.1 | 1295.1 | 1560.5 |
| LSZH | 192.3 | 275.5 | 206.6 | 291.2 |
| LTFM | 271.8 | 533.4 | 268.6 | 375.7 |

MAE:

| ADEP_mvt_key | catboost_residual | operational_aobt3 | catboost_direct | hierarchical_median |
|---|---:|---:|---:|---:|
| EDDF | 135.9 | 184.1 | 137.0 | 205.9 |
| EDDM | 136.5 | 248.2 | 138.3 | 219.4 |
| EGLL | 180.2 | 263.4 | 181.2 | 280.0 |
| EHAM | 123.5 | 200.7 | 124.3 | 148.9 |
| LEBL | 149.1 | 198.6 | 151.4 | 220.3 |
| LEMD | 134.9 | 169.3 | 138.4 | 203.0 |
| LFPG | 183.1 | 252.4 | 187.9 | 254.8 |
| LIRF | 300.8 | 390.2 | 311.2 | 386.5 |
| LSZH | 131.3 | 187.6 | 140.9 | 197.2 |
| LTFM | 180.5 | 380.1 | 182.9 | 244.1 |

## 7. Impacto de los outliers

Las cuatro estrategias se aplican SOLO al bloque de entrenamiento; la validación se puntúa siempre contra el target original.
`rmse_outlier` es el RMSE en las filas de validación con target ≤ 0 o > 3 h (muy pocas, pero con errores de miles de segundos).

| model | strategy | rmse | mae | rmse_normal | mae_normal | rmse_outlier | n_pred_negative |
|---|---|---:|---:|---:|---:|---:|---:|
| catboost_direct | drop_gt3h | 443.9 | 163.3 | 258.6 | 160.4 | 22373.6 | 40 |
| catboost_direct | drop_nonpositive | 421.7 | 170.0 | 294.8 | 167.8 | 18457.8 | 28 |
| catboost_direct | original | 425.5 | 167.6 | 292.8 | 165.4 | 19000.6 | 48 |
| catboost_direct | winsor_p01_p999 | 458.8 | 163.0 | 261.2 | 159.8 | 23344.2 | 0 |
| catboost_residual | drop_gt3h | 468.4 | 164.2 | 271.5 | 160.9 | 23805.1 | 95 |
| catboost_residual | drop_nonpositive | 466.7 | 164.3 | 272.1 | 161.0 | 23659.4 | 65 |
| catboost_residual | original | 466.6 | 164.3 | 271.9 | 161.1 | 23659.5 | 108 |
| catboost_residual | winsor_p01_p999 | 467.9 | 164.2 | 272.7 | 160.9 | 23722.7 | 60 |
| hierarchical_median | drop_gt3h | 549.8 | 233.4 | 380.0 | 230.0 | 24385.8 | 38 |
| hierarchical_median | drop_nonpositive | 548.2 | 233.5 | 380.2 | 230.1 | 24245.6 | 0 |
| hierarchical_median | original | 548.2 | 233.5 | 380.2 | 230.1 | 24245.6 | 38 |
| hierarchical_median | winsor_p01_p999 | 548.4 | 233.5 | 379.7 | 230.1 | 24300.2 | 0 |
| operational_aobt3 | drop_gt3h | 565.0 | 249.9 | 415.1 | 246.6 | 23826.1 | 357 |
| operational_aobt3 | drop_nonpositive | 563.5 | 249.9 | 415.3 | 246.7 | 23682.2 | 319 |
| operational_aobt3 | original | 563.5 | 249.9 | 415.3 | 246.7 | 23682.2 | 357 |
| operational_aobt3 | winsor_p01_p999 | 563.7 | 249.9 | 414.8 | 246.7 | 23737.4 | 319 |

RMSE por aeropuerto y estrategia para `catboost_direct` (media de folds):

| ADEP_mvt_key | drop_gt3h | drop_nonpositive | original | winsor_p01_p999 |
|---|---:|---:|---:|---:|
| EDDF | 199.5 | 206.9 | 204.5 | 202.4 |
| EDDM | 208.3 | 209.9 | 219.5 | 207.8 |
| EGLL | 270.3 | 276.0 | 276.1 | 272.8 |
| EHAM | 179.5 | 189.4 | 184.6 | 181.1 |
| LEBL | 219.3 | 226.2 | 224.1 | 218.4 |
| LEMD | 190.0 | 199.0 | 195.8 | 189.3 |
| LFPG | 297.4 | 306.9 | 305.2 | 300.6 |
| LIRF | 1388.7 | 1274.7 | 1295.1 | 1450.1 |
| LSZH | 195.9 | 204.5 | 206.6 | 195.4 |
| LTFM | 264.4 | 272.2 | 268.6 | 263.7 |

## 10. Variables con más señal (importancia CatBoost, media de folds)

| feature | importance |
|---|---:|
| mvt_minus_aobt3 | 17.67 |
| ADEP_mvt | 16.88 |
| mvt_minus_sched | 15.74 |
| mvt_minus_eobt1 | 11.74 |
| RUNWAY_mvt | 5.69 |
| STAND_mvt | 4.53 |
| departures_per_airport_day | 4.11 |
| aobt3_minus_sched | 3.58 |
| ADES_mvt | 2.29 |
| AIRCRAFT_OPERATOR_flt | 2.17 |
| mvt_minus_iobt | 1.81 |
| hour_utc | 1.40 |
| has_aobt3 | 1.35 |
| WK_TBL_CAT_flt | 1.26 |
| departures_per_airport_runway_hour | 1.19 |
| MARKET_SEGMENT_flt | 1.11 |
| departures_per_airport_hour | 1.10 |
| AIRCRAFT_TYPE_mvt | 0.97 |
| mvt_minus_lobt | 0.90 |
| hour_sin | 0.81 |
| arrivals_per_airport_hour | 0.77 |
| FLIGHT_TYPE_flt | 0.70 |
| hour_cos | 0.63 |
| month_cos | 0.54 |
| month | 0.49 |
| day_of_week | 0.32 |
| month_sin | 0.18 |
| is_weekend | 0.06 |
| FLIGHT_RULE_mvt | 0.01 |

## 8. Modelo recomendado

Mejor RMSE medio de los 3 backtests: **catboost_direct**; mejor MAE medio: **catboost_residual**.

- **CatBoost directo** gana en RMSE en los tres folds (≈ 576 / 328 / 373 s frente a 705 / 481 / 459 s de la mediana
  jerárquica y 717 / 494 / 480 s de la baseline operacional) y en MAE (≈ 192 / 153 / 158 s frente a 247 / 226 / 227 s).
  Empata con el CatBoost residual en MAE y MedAE, pero el residual pierde 20–75 s de RMSE porque hereda los errores de
  `AOBT_3` cuando ésta es basura (filas extremas, LIRF, LTFM) y depende de un fallback para el ~1 % sin plan de vuelo.
- La mediana jerárquica supera a la baseline operacional en RMSE en los tres folds y solo pierde en MAE cuando se corrige el
  sesgo por aeropuerto (`operational_aobt3_biascorr`), es decir, `MVT − AOBT_3` a secas es un mal predictor puntual.
- Por aeropuerto (media de folds) el cuadro es más matizado: el **residual** gana en 8 de 10 aeropuertos por 1–14 s de RMSE
  (LSZH, EHAM, EDDM, EGLL…), y el **directo** gana en LTFM (269 vs 272) y sobre todo en LIRF (1295 vs 1479), que por sí solo
  decide el RMSE global. Ambos baten con holgura a las baselines (p. ej. LTFM: 533 operacional, 376 jerárquica, ≈270 CatBoost).
  Un ensamble (media de ambos) es la opción natural y se propone validar antes de la submission.
- Estrategia de outliers: para CatBoost directo, `original` y `drop_nonpositive` dan el mejor RMSE global (F1 576/582, F2 328/329,
  F3 373/354); `drop_gt3h` y `winsor` mejoran el MAE (−5 s) y el RMSE en el rango 0–3 h (−15 a −60 s) pero empeoran el RMSE global
  (+30 a +60 s) porque el modelo deja de aprender que ciertos patrones (`mvt_minus_sched` enorme, sin plan de vuelo) preceden a
  taxi-outs de horas. **Recomendación: `drop_nonpositive`** (equivalente a `original` salvo 388 filas de ruido de sensor) si la
  métrica del concurso es RMSE; `winsor_p01_p999` si fuese MAE.
- Robustez por aeropuerto: `drop_gt3h` y `winsor` ganan en **9 de los 10 aeropuertos** (5–10 s de RMSE) y solo pierden en LIRF
  (+90 a +155 s), que es el único que decide el RMSE global. Es decir, la elección de estrategia es en realidad una apuesta sobre
  si en el ranking LIRF seguirá teniendo taxi-outs de horas (registros dudosos de `BLOCK_TIME`); con el target original el modelo
  cubre esa apuesta a costa de 5–10 s en el resto.
- Predicciones negativas: 0–22 por fold en CatBoost (de 160–190 k); se recortarán a 0 en la submission.

## 9. Qué falta antes de generar una submission

1. Confirmar la métrica oficial del concurso (RMSE vs MAE) y elegir la estrategia de outliers en consecuencia.
2. Estabilizar el early stopping: en F1 el conjunto de parada (últimos 14 días de junio) contiene extremos y detiene el modelo en
   125–413 iteraciones; conviene fijar el nº de iteraciones a partir de los folds (≈ 700–1000) o usar MAE como métrica de parada.
3. Entrenar el modelo final con los 12 meses de 2025 (`python src/train_baseline.py --models cb_direct hier`) y validar con
   un backtest adicional de horizonte largo (p. ej. ene–may → dic) porque el ranking está 1 y 7 meses después del training.
4. Escribir `src/make_submission.py`: `load_features("rank")` → predecir → recortar a ≥ 0 → redondear a `Int32` → alinear por
   `MVT_ID_mvt` con `submitting.parquet` (mismo orden, 344 841 filas) → comprobar que no hay nulos ni no-finitos → guardar en
   `submissions/` (excluido de git). Sin subir nada al bucket hasta validar el archivo.
5. Mejoras con más recorrido que el ajuste fino: features de congestión en ventanas de ±15/30 min (no solo por hora), cola de
   salidas de la misma pista en los 10 min anteriores, target encoding temporal de `STAND` y `FLIGHT_mvt`, y un ensamble
   directo + residual (media) que en los folds recortaría el RMSE unos pocos segundos sin coste.
6. Tratar LIRF aparte (RMSE 4–5× el resto): revisar si sus taxi-outs > 2 h son errores de `BLOCK_TIME`; si el concurso los puntúa,
   un modelo específico o un término de riesgo para LIRF puede valer más que cualquier otra mejora.
