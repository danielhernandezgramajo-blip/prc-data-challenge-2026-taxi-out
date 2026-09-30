# V3 (etapa 2 de V2) — Informe de experimentos (mismo protocolo de origen creciente, 8 folds)

Generado 2026-09-30 16:40 UTC. Holdout de diciembre abierto (segunda apertura): **True**. Finalista congelada antes de abrirlo: **S2A_mix** (2026-09-29T23:45:59+00:00).

## 1. Motivación: dónde estaba el error restante de S_mix (desarrollo, mayo–noviembre)

Descomposición del error cuadrático de S_mix por clase de régimen: clase 0 (taxi físico) 59 %, clase 1 (bloque = hora
programada) 16 % y **clase 2 (fecha del bloque desplazada un día) 25 % con solo 10 filas**. Un clasificador perfecto bajaría el
RMSE pooled de 321 a 231 s. Mecanismo de la clase 2 (LIRF, sin AOBT_3, fecha de MVT posterior a la programada): el off-block se
registra con la hora correcta pero con la fecha del programado. En 2025 esa celda tiene 68 filas: 54 de clase 1, 11 de clase 2 y
3 normales; el clasificador de V2 les asignaba p2 ≈ 0,005 (predicción 30–60 k s frente a 87 k reales).

## 2. Qué se probó (registro previo a los resultados) y regla de decisión

| candidate | family | definition | eligible |
|---|---|---|---|
| V1P | referencia | 0.5*direct + 0.5*residual (v1_protocol) | False |
| S_mix | referencia (candidate_v2) | proba V2; normal_v2 = 0.5*direct_normal + 0.5*residual_V1 | False |
| S2_mix_a0 | S2 (ablación) | proba2 alpha=0 (feature days_after_sched, sin celdas) + normal_v2 | False |
| S2_mix_a5 | S2 (sensibilidad) | proba2 alpha=5 + normal_v2 | False |
| S2_mix | S2 (primario) | proba2 alpha=20 + normal_v2 | True |
| S2_mix_a100 | S2 (sensibilidad) | proba2 alpha=100 + normal_v2 | False |
| A_mix | A (superficie) | proba V2 + normal_s = 0.5*direct_normal_s + 0.5*residual_normal_s | True |
| A_mix_dn | A (información) | proba V2 + direct_normal_s | False |
| A_mix_rn | A (información) | proba V2 + residual_normal_s | False |
| A_mix_4 | A (superficie) | proba V2 + 0.25*(direct_normal + residual_V1 + direct_normal_s + residual_normal_s) | True |
| S2A_mix | S2+A | proba2 alpha=20 + normal_s | True |
| S2A_mix_4 | S2+A | proba2 alpha=20 + normal 4 componentes | True |
| S3_mix | S3 (2c, post-hoc) | proba2 celdas jerárquicas por banda de retraso (>=12 h), alpha=20 + normal_v2 | False |
| S3A_mix | S3+A (2c, post-hoc) | proba2 celdas jerárquicas por banda (>=12 h), alpha=20 + normal_s | False |
| S3A_mix_4 | S3+A (2c, post-hoc) | proba2 celdas jerárquicas por banda (>=12 h), alpha=20 + normal 4 componentes | False |
| S4_mix | S4 (2d, hipótesis post-hoc, umbral anidado) | proba2 celdas jerárquicas con umbral elegido por LOO en el bloque de entrenamiento (o plana), alpha=20 + normal_v2 | False |
| S4A_mix | S4+A (2d, hipótesis post-hoc, umbral anidado) | proba2 celdas jerárquicas con umbral anidado, alpha=20 + normal_s | False |
| S4A_mix_4 | S4+A (2d, hipótesis post-hoc, umbral anidado) | proba2 celdas jerárquicas con umbral anidado, alpha=20 + normal 4 componentes | False |

menor RMSE pooled de desarrollo entre las elegibles que cumplen todos los criterios requeridos frente a S_mix y V1P. α = 20 prefijado (0 = ablación sin celdas; 5 y 100 = sensibilidad, no se selecciona entre ellos).

## 3. Cómo se evitó el leakage

- Mismos cortes por `MVT_TIME_UTC_mvt`, misma estrategia (drop_nonpositive) y mismos ids/orden de validación que V2 (verificado).
- `days_after_sched` = fecha(MVT) − fecha(SCHED): sin target ni `BLOCK_TIME`; ambas columnas están en ranking.
- Las cuentas de las celdas y la tabla de clase 2 se calculan SOLO con el bloque de entrenamiento de cada fold.
- Features de congestión (etapa 2b) construidas solo con marcas de tiempo/claves; congestión relativa ajustada con el bloque de entrenamiento.
- Validación puntuada contra el target original completo; diciembre sellado hasta congelar la finalista.

| variable | usada_en | usa_target | usa_BLOCK_TIME | disponible_en_ranking | estado |
|---|---|---|---|---|---|
| month | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| day_of_week | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| hour_utc | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| is_weekend | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| hour_sin | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| hour_cos | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| month_sin | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| month_cos | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| mvt_minus_aobt3 | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| mvt_minus_eobt1 | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| mvt_minus_lobt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| mvt_minus_iobt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| mvt_minus_sched | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| aobt3_minus_sched | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| has_aobt3 | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| departures_per_airport_hour | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| departures_per_airport_day | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| departures_per_airport_runway_hour | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| arrivals_per_airport_hour | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| ADEP_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| ADES_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| RUNWAY_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| STAND_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| AIRCRAFT_TYPE_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| MARKET_SEGMENT_flt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| FLIGHT_TYPE_flt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| WK_TBL_CAT_flt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| AIRCRAFT_OPERATOR_flt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| FLIGHT_RULE_mvt | clasificador etapa 2, direct_normal_s, residual_normal_s (+ componentes V2 reutilizados) | False | False | True | usada |
| days_after_sched | clasificador etapa 2 (feature) y clave de celdas/tabla de clase 2 | False | False | True | usada |
| celdas (aeropuerto | desplazada ≥ 1 día, sin AOBT_3): n, k1, k2 | actualización bayesiana de p1/p2 (α = 20) | True | False | False | tabla ajustada en entrenamiento (como el codec o las medianas) |
| dep_pm15 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| dep_pm30 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| rwy_prev10 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| rwy_prev15 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| rwy_prev30 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| arr_pm15 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| arr_pm30 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| net_pm15 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| net_pm30 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| min_since_prev_rwy_dep | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| min_to_next_rwy_sched | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| rel_dep_pm30 | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| rel_dep_hour | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |
| apt_hour_dow | direct_normal_s, residual_normal_s (A_surface) | False | False | True | usada (etapa 2b) |

## 4. Unidades ejecutadas (tiempos, AUC del clasificador, RMSE de los componentes)

| variant | fold | model | fit_seconds | n_train | logloss | auc_class1 | auc_class2 | rmse | rmse_normal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_surface | Aug | direct_normal | 260.2 | 1137610 |  |  |  | 419.3 | 278.3 |
| A_surface | Aug | residual_normal | 263.1 | 1137610 |  |  |  | 426.0 | 283.0 |
| A_surface | Dec | direct_normal | 448.9 | 1827079 |  |  |  | 380.0 | 252.7 |
| A_surface | Dec | residual_normal | 449.2 | 1827079 |  |  |  | 385.9 | 257.7 |
| A_surface | Jul | direct_normal | 215.4 | 954937 |  |  |  | 645.2 | 318.5 |
| A_surface | Jul | residual_normal | 211.3 | 954937 |  |  |  | 624.0 | 325.4 |
| A_surface | Jun | direct_normal | 179.8 | 780079 |  |  |  | 447.5 | 279.7 |
| A_surface | Jun | residual_normal | 171.1 | 780079 |  |  |  | 452.9 | 284.8 |
| A_surface | May | direct_normal | 139.8 | 604064 |  |  |  | 344.2 | 241.9 |
| A_surface | May | residual_normal | 129.4 | 604064 |  |  |  | 348.4 | 245.5 |
| A_surface | Nov | residual_normal | 410.4 | 1672965 |  |  |  | 407.2 | 243.3 |
| A_surface | Oct | direct_normal | 363.5 | 1495709 |  |  |  | 283.4 | 259.1 |
| A_surface | Oct | residual_normal | 326.7 | 1495709 |  |  |  | 288.0 | 264.7 |
| A_surface | Sep | direct_normal | 311.5 | 1319766 |  |  |  | 450.0 | 261.5 |
| A_surface | Sep | residual_normal | 309.3 | 1319766 |  |  |  | 456.5 | 266.3 |
| S2_mixture | Aug | proba_a0 | 159.2 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Aug | proba_a100 | 160.7 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Aug | proba_a20 | 160.1 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Aug | proba_a5 | 159.7 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Aug | proba_b20 | 161.2 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Aug | proba_n20 | 161.7 | 1195968 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Dec | proba_a0 | 239.4 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Dec | proba_a100 | 240.6 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Dec | proba_a20 | 240.2 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Dec | proba_a5 | 239.8 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Dec | proba_b20 | 241.1 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Dec | proba_n20 | 241.6 | 1919005 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_a0 | 124.9 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_a100 | 126.5 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_a20 | 125.9 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_a5 | 125.4 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_b20 | 127.0 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jul | proba_n20 | 127.5 | 1005290 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_a0 | 101.3 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_a100 | 102.7 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_a20 | 102.2 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_a5 | 101.8 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_b20 | 103.2 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Jun | proba_n20 | 103.7 | 822210 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_a0 | 96.9 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_a100 | 98.4 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_a20 | 97.8 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_a5 | 97.3 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_b20 | 98.9 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | May | proba_n20 | 99.4 | 637029 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_a0 | 220.9 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_a100 | 222.2 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_a20 | 221.8 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_a5 | 221.4 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_b20 | 222.7 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Nov | proba_n20 | 223.1 | 1756691 | 0.2 | 0.9 | 1.0 |  |  |
| S2_mixture | Oct | proba_a0 | 195.2 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Oct | proba_a100 | 196.6 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Oct | proba_a20 | 196.1 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Oct | proba_a5 | 195.7 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Oct | proba_b20 | 197.0 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Oct | proba_n20 | 197.5 | 1571037 | 0.1 | 0.9 | nan |  |  |
| S2_mixture | Sep | proba_a0 | 176.5 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Sep | proba_a100 | 178.0 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Sep | proba_a20 | 177.5 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Sep | proba_a5 | 177.0 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Sep | proba_b20 | 178.5 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |
| S2_mixture | Sep | proba_n20 | 179.0 | 1387105 | 0.1 | 0.9 | 1.0 |  |  |

## 5. Resultados pooled de desarrollo (RMSE sobre todas las filas, no media de medias)

| candidate | rmse | mae | medae | cov_le300 | cov_le600 | rmse_class0 | rmse_class1 | rmse_class2 | sse_share_class0 | sse_share_class1 | sse_share_class2 | n_pred_negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P | 390.6 | 161.1 | 111.8 | 87.71 | 97.76 | 248.7 | 977.4 | 79747.6 | 38.7 | 28.8 | 32.5 | 103 |
| S_mix | 320.9 | 159.3 | 111.4 | 87.80 | 97.80 | 253.1 | 590.2 | 57574.1 | 59.3 | 15.6 | 25.1 | 69 |
| S2_mix_a0 | 319.9 | 159.3 | 111.4 | 87.80 | 97.80 | 250.9 | 588.0 | 58137.1 | 58.7 | 15.5 | 25.8 | 68 |
| S2_mix | 302.5 | 159.1 | 111.4 | 87.80 | 97.80 | 251.3 | 530.0 | 48509.3 | 65.8 | 14.1 | 20.1 | 68 |
| A_mix | 319.1 | 157.1 | 109.4 | 88.17 | 97.86 | 248.0 | 611.4 | 57650.2 | 57.7 | 16.9 | 25.5 | 73 |
| A_mix_4 | 319.3 | 157.6 | 110.0 | 88.08 | 97.85 | 249.7 | 600.1 | 57612.0 | 58.4 | 16.2 | 25.4 | 65 |
| S2A_mix | 300.9 | 156.8 | 109.4 | 88.16 | 97.86 | 246.6 | 555.1 | 48546.2 | 64.1 | 15.6 | 20.3 | 74 |
| S2A_mix_4 | 301.0 | 157.4 | 110.0 | 88.07 | 97.84 | 248.1 | 541.7 | 48527.6 | 64.8 | 14.9 | 20.3 | 65 |
| S3_mix | 298.7 | 159.0 | 111.4 | 87.80 | 97.80 | 251.3 | 517.7 | 46256.8 | 67.5 | 13.8 | 18.7 | 68 |
| S3A_mix | 297.1 | 156.8 | 109.4 | 88.16 | 97.86 | 246.5 | 543.2 | 46288.6 | 65.7 | 15.4 | 18.9 | 74 |
| S3A_mix_4 | 297.1 | 157.3 | 110.0 | 88.07 | 97.84 | 248.0 | 529.6 | 46272.6 | 66.5 | 14.6 | 18.9 | 65 |
| S4_mix | 299.5 | 159.0 | 111.4 | 87.80 | 97.80 | 251.3 | 518.8 | 46788.4 | 67.2 | 13.8 | 19.0 | 68 |
| S4A_mix | 297.8 | 156.8 | 109.4 | 88.16 | 97.86 | 246.5 | 544.3 | 46818.7 | 65.4 | 15.4 | 19.3 | 74 |
| S4A_mix_4 | 297.9 | 157.3 | 110.0 | 88.07 | 97.84 | 248.0 | 530.7 | 46803.4 | 66.1 | 14.6 | 19.3 | 65 |

Sensibilidad a α (S2_mix): 

| candidate | rmse | mae | cov_le300 | rmse_class1 | rmse_class2 |
|---|---:|---:|---:|---:|---:|
| S_mix | 320.9 | 159.3 | 87.80 | 590.2 | 57574.1 |
| S2_mix_a0 | 319.9 | 159.3 | 87.80 | 588.0 | 58137.1 |
| S2_mix_a5 | 300.0 | 159.1 | 87.80 | 546.7 | 45073.1 |
| S2_mix | 302.5 | 159.1 | 87.80 | 530.0 | 48509.3 |
| S2_mix_a100 | 310.6 | 159.2 | 87.80 | 546.3 | 53840.7 |

## 6. Por fold (referencias y elegibles)

| fold | V1P | S_mix | S2_mix_a0 | S2_mix | A_mix | A_mix_4 | S2A_mix | S2A_mix_4 | S3_mix | S3A_mix | S3A_mix_4 | S4_mix | S4A_mix | S4A_mix_4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aug | 357.2 | 293.2 | 295.0 | 284.0 | 290.0 | 290.6 | 282.0 | 282.0 | 283.6 | 281.5 | 281.6 | 283.6 | 281.5 | 281.6 |
| Jul | 559.0 | 415.6 | 404.9 | 362.3 | 414.9 | 414.5 | 361.3 | 360.9 | 351.8 | 350.7 | 350.3 | 351.8 | 350.7 | 350.3 |
| Jun | 403.7 | 344.6 | 344.2 | 330.6 | 343.4 | 343.4 | 329.4 | 329.4 | 329.6 | 328.4 | 328.4 | 329.6 | 328.4 | 328.4 |
| May | 324.7 | 316.5 | 315.7 | 316.1 | 314.9 | 315.1 | 314.6 | 314.8 | 315.8 | 314.4 | 314.5 | 315.8 | 314.4 | 314.5 |
| Nov | 365.5 | 300.4 | 300.0 | 257.9 | 298.4 | 298.7 | 255.7 | 256.1 | 245.9 | 243.6 | 244.0 | 252.2 | 250.0 | 250.4 |
| Oct | 253.9 | 246.4 | 245.8 | 247.3 | 244.3 | 244.4 | 245.7 | 245.6 | 245.7 | 244.1 | 244.0 | 246.2 | 244.6 | 244.5 |
| Sep | 394.2 | 299.2 | 306.7 | 297.0 | 296.2 | 296.9 | 294.6 | 295.1 | 295.6 | 293.2 | 293.6 | 295.7 | 293.3 | 293.7 |

## 7. Por aeropuerto (pooled de desarrollo; RMSE)

| ADEP_mvt | S_mix | S2A_mix |
|---|---:|---:|
| EDDF | 189.6 | 187.6 |
| EDDM | 180.6 | 179.5 |
| EGLL | 259.7 | 258.0 |
| EHAM | 173.0 | 172.2 |
| LEBL | 231.1 | 228.1 |
| LEMD | 196.1 | 195.1 |
| LFPG | 278.4 | 270.5 |
| LIRF | 808.7 | 713.4 |
| LSZH | 353.0 | 351.0 |
| LTFM | 257.7 | 255.9 |

## 8. Bootstrap pareado por bloques aeropuerto-día (1 000 remuestreos, SEED 42)

| candidate | vs | rmse_baseline | rmse_candidate | delta_rmse_point | delta_p2.5 | delta_p97.5 | p_improve | n_blocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1P | S_mix | 320.9 | 390.6 | 69.7 | 42.6 | 93.7 | 0.000 | 2140 |
| S2_mix_a0 | S_mix | 320.9 | 319.9 | -1.0 | -4.1 | 2.0 | 0.735 | 2140 |
| S2_mix | S_mix | 320.9 | 302.5 | -18.4 | -29.3 | -7.1 | 1.000 | 2140 |
| A_mix | S_mix | 320.9 | 319.1 | -1.8 | -2.5 | -1.3 | 1.000 | 2140 |
| A_mix_4 | S_mix | 320.9 | 319.3 | -1.6 | -2.1 | -1.3 | 1.000 | 2140 |
| S2A_mix | S_mix | 320.9 | 300.9 | -20.0 | -31.2 | -8.8 | 1.000 | 2140 |
| S2A_mix_4 | S_mix | 320.9 | 301.0 | -19.9 | -31.0 | -8.7 | 1.000 | 2140 |
| S3_mix | S_mix | 320.9 | 298.7 | -22.2 | -35.6 | -8.8 | 1.000 | 2140 |
| S3A_mix | S_mix | 320.9 | 297.1 | -23.8 | -37.2 | -10.5 | 1.000 | 2140 |
| S3A_mix_4 | S_mix | 320.9 | 297.1 | -23.8 | -37.2 | -10.3 | 1.000 | 2140 |
| S4_mix | S_mix | 320.9 | 299.5 | -21.4 | -34.3 | -8.4 | 1.000 | 2140 |
| S4A_mix | S_mix | 320.9 | 297.8 | -23.1 | -36.1 | -10.3 | 1.000 | 2140 |
| S4A_mix_4 | S_mix | 320.9 | 297.9 | -23.0 | -35.9 | -10.1 | 1.000 | 2140 |
| S_mix | V1P | 390.6 | 320.9 | -69.7 | -93.7 | -42.6 | 1.000 | 2140 |
| S2_mix_a0 | V1P | 390.6 | 319.9 | -70.7 | -95.4 | -43.2 | 1.000 | 2140 |
| S2_mix | V1P | 390.6 | 302.5 | -88.0 | -119.0 | -51.1 | 1.000 | 2140 |
| A_mix | V1P | 390.6 | 319.1 | -71.5 | -95.3 | -44.5 | 1.000 | 2140 |
| A_mix_4 | V1P | 390.6 | 319.3 | -71.3 | -95.1 | -44.3 | 1.000 | 2140 |
| S2A_mix | V1P | 390.6 | 300.9 | -89.7 | -120.7 | -53.3 | 1.000 | 2140 |
| S2A_mix_4 | V1P | 390.6 | 301.0 | -89.6 | -120.6 | -53.0 | 1.000 | 2140 |
| S3_mix | V1P | 390.6 | 298.7 | -91.8 | -124.2 | -52.7 | 1.000 | 2140 |
| S3A_mix | V1P | 390.6 | 297.1 | -93.5 | -125.7 | -54.5 | 1.000 | 2140 |
| S3A_mix_4 | V1P | 390.6 | 297.1 | -93.4 | -125.6 | -54.4 | 1.000 | 2140 |
| S4_mix | V1P | 390.6 | 299.5 | -91.1 | -123.9 | -52.6 | 1.000 | 2140 |
| S4A_mix | V1P | 390.6 | 297.8 | -92.8 | -125.3 | -54.3 | 1.000 | 2140 |
| S4A_mix_4 | V1P | 390.6 | 297.9 | -92.7 | -125.5 | -54.2 | 1.000 | 2140 |

## 9. Criterios de selección (solo desarrollo; frente a S_mix y frente a V1P)

| candidate | vs | rmse_gain_s | cov5_gain_pp | folds_not_worse | worst_fold_worsening_pct | min_cov5_fold | rmse_normal_gain_s | airports_worse_of_10 | share_gain_top50_rows | p_improve_boot | all_required_criteria | c8_no_harm_bulk | c8_old_share_top50 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| V1P | S_mix | -69.7 | -0.09 | 0/7 | 34.5 | 85.4 | -2.9 | 10 | nan | 0.000 | False | False | True |
| S2_mix_a0 | S_mix | 1.0 | 0.00 | 5/7 | 2.5 | 85.4 | 1.1 | 7 | 4.017 | 0.735 | False | False | False |
| S2_mix | S_mix | 18.4 | 0.00 | 6/7 | 0.4 | 85.4 | -0.2 | 8 | 1.135 | 1.000 | False | False | False |
| A_mix | S_mix | 1.8 | 0.36 | 7/7 | -0.2 | 85.8 | 2.5 | 0 | 0.627 | 1.000 | False | True | True |
| A_mix_4 | S_mix | 1.6 | 0.27 | 7/7 | -0.3 | 85.7 | 2.1 | 0 | 0.395 | 1.000 | False | True | True |
| S2A_mix | S_mix | 20.0 | 0.36 | 7/7 | -0.3 | 85.7 | 1.8 | 0 | 1.061 | 1.000 | True | True | False |
| S2A_mix_4 | S_mix | 19.9 | 0.27 | 7/7 | -0.3 | 85.7 | 1.7 | 0 | 1.053 | 1.000 | True | True | False |
| S3_mix | S_mix | 22.2 | 0.00 | 7/7 | -0.2 | 85.4 | 0.4 | 8 | 1.077 | 1.000 | False | False | False |
| S3A_mix | S_mix | 23.8 | 0.36 | 7/7 | -0.7 | 85.7 | 2.4 | 0 | 1.018 | 1.000 | True | True | False |
| S3A_mix_4 | S_mix | 23.8 | 0.27 | 7/7 | -0.6 | 85.7 | 2.3 | 0 | 1.011 | 1.000 | True | True | False |
| S4_mix | S_mix | 21.4 | 0.00 | 7/7 | -0.1 | 85.4 | 0.4 | 8 | 1.083 | 1.000 | False | False | False |
| S4A_mix | S_mix | 23.1 | 0.36 | 7/7 | -0.7 | 85.7 | 2.3 | 0 | 1.021 | 1.000 | True | True | False |
| S4A_mix_4 | S_mix | 23.0 | 0.27 | 7/7 | -0.6 | 85.7 | 2.2 | 0 | 1.014 | 1.000 | True | True | False |
| S_mix | V1P | 69.7 | 0.09 | 7/7 | -2.5 | 85.4 | 2.9 | 0 | 0.939 | 1.000 | True | True | False |
| S2_mix_a0 | V1P | 70.7 | 0.09 | 7/7 | -2.8 | 85.4 | 4.1 | 0 | 0.934 | 1.000 | True | True | False |
| S2_mix | V1P | 88.0 | 0.09 | 7/7 | -2.6 | 85.4 | 2.8 | 0 | 0.960 | 1.000 | True | True | False |
| A_mix | V1P | 71.5 | 0.45 | 7/7 | -3.0 | 85.8 | 5.4 | 0 | 0.914 | 1.000 | True | True | False |
| A_mix_4 | V1P | 71.3 | 0.36 | 7/7 | -3.0 | 85.7 | 5.0 | 0 | 0.918 | 1.000 | True | True | False |
| S2A_mix | V1P | 89.7 | 0.45 | 7/7 | -3.1 | 85.7 | 4.7 | 0 | 0.944 | 1.000 | True | True | False |
| S2A_mix_4 | V1P | 89.6 | 0.36 | 7/7 | -3.1 | 85.7 | 4.6 | 0 | 0.945 | 1.000 | True | True | False |
| S3_mix | V1P | 91.8 | 0.09 | 7/7 | -2.7 | 85.4 | 3.3 | 0 | 0.953 | 1.000 | True | True | False |
| S3A_mix | V1P | 93.5 | 0.45 | 7/7 | -3.2 | 85.7 | 5.3 | 0 | 0.938 | 1.000 | True | True | False |
| S3A_mix_4 | V1P | 93.4 | 0.36 | 7/7 | -3.1 | 85.7 | 5.2 | 0 | 0.939 | 1.000 | True | True | False |
| S4_mix | V1P | 91.1 | 0.09 | 7/7 | -2.7 | 85.4 | 3.3 | 0 | 0.954 | 1.000 | True | True | False |
| S4A_mix | V1P | 92.8 | 0.45 | 7/7 | -3.2 | 85.7 | 5.2 | 0 | 0.939 | 1.000 | True | True | False |
| S4A_mix_4 | V1P | 92.7 | 0.36 | 7/7 | -3.1 | 85.7 | 5.1 | 0 | 0.940 | 1.000 | True | True | False |

Decisión en desarrollo (regla prefijada): elegibles ['S2_mix', 'A_mix', 'A_mix_4', 'S2A_mix', 'S2A_mix_4']; cumplen ['S2A_mix', 'S2A_mix_4']; finalista **S2A_mix**.


### 9b. Etapa 2c (post-hoc, definida DESPUÉS de abrir diciembre)

Al abrir diciembre con la finalista congelada se vio que la celda plana reparte la masa de clase 2 (p2 ≈ 0,13–0,16) también a
filas de retraso corto de la celda, que casi nunca son de clase 2 (en 2025: 1 de 43 con retraso < 12 h frente a 10 de 25 con
retraso ≥ 12 h; test exacto de Fisher p < 0,001), y además el prior del clasificador deja masa en p0 en una banda donde no hay
filas normales. La versión jerárquica (S3*, `proba_b20`) añade dentro de la celda una segunda actualización bayesiana por banda de
retraso (≥ 12 h) con el mismo α = 20. **Esta variante no estaba registrada antes de abrir diciembre**, por lo que diciembre no es
una validación limpia para ella; se evalúa con los mismos criterios en desarrollo y se documenta la elección en `final_choice.json`.


### 9c. Etapa 2d: umbral elegido dentro de cada ventana de entrenamiento (validación anidada)

Para no fijar el umbral de 12 h a mano (elegido tras ver diciembre), S4* elige el umbral en cada fold **solo con las filas de
celda del bloque de entrenamiento**: log-loss leave-one-out de la distribución de clases para {celda plana} ∪ {banda ≥ h, h ∈
6/9/12/18/24 h}, con el mismo α = 20; si ningún umbral mejora a la celda plana, se usa la celda plana. El mes de validación y
diciembre nunca intervienen en la elección. Lo que sigue siendo post-hoc es la HIPÓTESIS (que la clase 2 dependa del retraso),
nacida de mirar diciembre y el año completo; eso no lo cura ninguna validación con estos datos, solo el ranking 2026.

| fold | n_celda_train | umbral_elegido_h | flat | band_6h | band_9h | band_12h | band_18h | band_24h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| May | 83 | 12 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Jun | 94 | 12 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Jul | 111 | 12 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Aug | 148 | 12 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Sep | 175 | 18 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Oct | 201 | 18 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Nov | 219 | 18 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Dec | 236 | 12 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |

### 9d. Criterio c8 redefinido

El c8 original de V2 ("la ganancia no depende de pocas filas": < 90 % de la reducción de SSE en las 50 mejores filas) es
inaplicable por construcción a una corrección de un régimen raro: cualquier arreglo de la cola lo incumple aunque sea real.
En V3 se sustituye por **c8_no_harm_bulk**: el RMSE de la población normal (0 < y ≤ 3 h) no empeora más de 1 s y a lo sumo 2 de
los 10 aeropuertos empeoran en RMSE pooled. El valor antiguo se sigue reportando (`c8_old_share_top50`) como información.


Elección final para `candidate_v3`: **S4A_mix** (post-hoc: True; finalista congelada por la regla prefijada: S2A_mix). Motivo: Hipótesis post-hoc (bandas de retraso dentro de la celda LIRF sin AOBT_3, nacida al abrir diciembre) con el parámetro elegido de forma anidada: el umbral (6/9/12/18/24 h o celda plana) se selecciona por log-loss leave-one-out SOLO con las filas de celda del bloque de entrenamiento de cada fold; diciembre nunca interviene en la elección del parámetro. Cumple todos los criterios requeridos en desarrollo (incl. c8 redefinido: no daña a la población normal ni a los aeropuertos) frente a S_mix y V1P: pooled 320,9 → 297,8 s, 7/7 folds, P(mejora)=1,000 (IC −36/−10). S3A_mix (umbral fijo 12 h elegido tras ver diciembre) queda como experimental; S2A_mix es la finalista rigurosamente defendible por la regla prefijada (300,9). Diciembre (234,4) no es validación limpia para S4A.


### Escalera de riesgo para la subida

| Candidata | Estado metodológico | Qué apuesta |
|---|---|---|
| V1 (`candidate_v1`) | protocolo V1, sin mezcla | no apuesta por el artefacto BLOCK = SCHED; peor si persiste (lo esperable) |
| V2 S_mix (`candidate_v2`) | congelada y confirmada en diciembre (primera apertura) | apuesta por el artefacto; celdas sin tratar |
| S2A_mix | finalista por la regla prefijada de la etapa 2; diciembre = segunda apertura | + celdas planas + congestión; mejora confirmada |
| S3A_mix | post-hoc (umbral fijo tras ver diciembre) | + banda de retraso fija: experimental |
| S4A_mix (`candidate_v3`) | hipótesis post-hoc, parámetro anidado | + banda con umbral elegido en entrenamiento: experimental, la mejor apuesta |

## 10. Holdout de diciembre (segunda apertura; confirmación, no selección)

Diciembre ya se había abierto en la etapa 1 para comparar S_mix con V1P; por eso aquí NO puede considerarse un holdout
intacto. Se abre una segunda vez, una sola, con la finalista congelada por la regla prefijada, y se reporta como confirmación.

| candidate | rmse | mae | medae | cov_le300 | cov_le600 | rmse_class0 | rmse_class1 | rmse_class2 | n_pred_negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P | 332.2 | 152.4 | 107.6 | 88.82 | 98.12 | 228.8 | 718.0 | 76419.1 | 17 |
| S_mix | 241.1 | 150.6 | 107.6 | 88.86 | 98.15 | 229.3 | 327.2 | 21924.1 | 2 |
| S2_mix_a0 | 240.5 | 150.5 | 107.6 | 88.88 | 98.15 | 229.4 | 327.5 | 20546.8 | 2 |
| S2_mix_a5 | 242.4 | 150.5 | 107.6 | 88.88 | 98.15 | 234.2 | 362.3 | 4932.2 | 2 |
| S2_mix | 241.2 | 150.5 | 107.6 | 88.88 | 98.15 | 233.2 | 352.9 | 7309.8 | 2 |
| S2_mix_a100 | 239.5 | 150.5 | 107.6 | 88.88 | 98.15 | 231.2 | 336.6 | 12599.8 | 2 |
| A_mix | 237.6 | 148.0 | 105.4 | 89.28 | 98.21 | 223.7 | 347.5 | 21995.3 | 2 |
| A_mix_dn | 238.1 | 149.3 | 106.3 | 89.06 | 98.18 | 224.4 | 346.8 | 21904.1 | 1 |
| A_mix_rn | 241.9 | 149.4 | 105.9 | 89.12 | 98.11 | 228.3 | 349.7 | 22086.5 | 15 |
| A_mix_4 | 238.5 | 148.7 | 106.0 | 89.18 | 98.20 | 225.6 | 336.6 | 21959.7 | 2 |
| S2A_mix | 237.8 | 148.0 | 105.4 | 89.28 | 98.20 | 228.0 | 372.1 | 7367.4 | 2 |
| S2A_mix_4 | 238.6 | 148.7 | 106.0 | 89.19 | 98.19 | 229.7 | 361.9 | 7338.6 | 2 |
| S3_mix | 237.9 | 150.5 | 107.6 | 88.88 | 98.15 | 230.5 | 348.3 | 3810.6 | 2 |
| S3A_mix | 234.4 | 147.9 | 105.4 | 89.28 | 98.20 | 225.2 | 367.9 | 3897.3 | 2 |
| S3A_mix_4 | 235.3 | 148.6 | 106.0 | 89.19 | 98.19 | 227.0 | 357.5 | 3854.0 | 2 |
| S4_mix | 237.9 | 150.5 | 107.6 | 88.88 | 98.15 | 230.5 | 348.3 | 3810.6 | 2 |
| S4A_mix | 234.4 | 147.9 | 105.4 | 89.28 | 98.20 | 225.2 | 367.9 | 3897.3 | 2 |
| S4A_mix_4 | 235.3 | 148.6 | 106.0 | 89.19 | 98.19 | 227.0 | 357.5 | 3854.0 | 2 |

## 11. Decisión

**crear candidate_v3 (S4A_mix: hipótesis post-hoc con parámetro anidado)** — Etapa 2 sobre V2. Finalista rigurosa por la regla prefijada: S2A_mix (celdas planas + congestión; desarrollo pooled 320,9 → 300,9 s; 7/7 folds; P(mejora)=1,000; cumple c1–c6, bootstrap y el c8 redefinido). S3A_mix (umbral fijo 12 h elegido tras abrir diciembre) se etiqueta EXPERIMENTAL/post-hoc. S4A_mix repite la hipótesis de bandas con el umbral elegido por log-loss leave-one-out dentro de cada ventana de entrenamiento (May–Aug y Dec: 12 h; Sep–Nov: 18 h; todo 2025: 12 h): desarrollo pooled 297,8 s (−23,1 vs S_mix; 7/7 folds; P(mejora)=1,000; IC −36/−10), cumple todos los criterios requeridos frente a S_mix y V1P. Diciembre (234,4) es segunda apertura y no valida la hipótesis. Con todo 2025 el umbral anidado coincide con 12 h, por lo que candidate_v3 es byte-idéntica a la generada con S3A_mix (misma sha256). V1 y candidate_v2 intactas. Nada subido al bucket.


`submissions/candidate_v3.parquet` · sha256 `165c6e298a033d2ad96765e3ff3652615f20d6411a15ad6acd144b76ef920ee0` · V1 intacta: `d482a42b3230d7e9eeba3890a66fc5ab4b9362d5e54a33c0151a265cfb5c4294` · V2 intacta: `e1db051a419e18d7314b56fb5b8105c93170f99c4e4254b1aeab7f8fc7f84752`
