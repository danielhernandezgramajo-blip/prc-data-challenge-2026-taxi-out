# V2 — Informe de experimentos (protocolo de origen creciente, 8 folds)

Generado 2026-09-29 16:45 UTC. Holdout de diciembre abierto: **True**. Finalista congelada antes de abrirlo: **S_mix** (2026-09-29T12:00:00Z).

## 1. Qué se probó y qué no

| candidate | family | definition | fits_reused |
|---|---|---|---|
| V1P | baseline | 0.5*direct + 0.5*residual (v1_protocol, features V1, 1000 it., drop_nonpositive) | v1_protocol |
| V1P_direct | baseline-component | catboost directo v1_protocol | v1_protocol |
| V1P_residual | baseline-component | catboost residual v1_protocol (fallback mediana jerárquica sin AOBT_3) | v1_protocol |
| S_mix | S (mezcla por régimen) | p0*normal + p1*(MVT-SCHED) + p2*(86400+normal); normal = 0.5*direct_normal + 0.5*residual | S_mixture + v1_protocol.residual |
| S_mix_dn | S (mezcla por régimen) | como S_mix con normal = direct_normal | S_mixture |
| S_mix_onV1P | S (control) | como S_mix con normal = V1P (regresor no depurado de extremos) | S_mixture.proba + v1_protocol |
| S_mix_floor | S + post-proceso | S_mix con suelo pred >= (MVT-AOBT_3) - 3600 s si hay AOBT_3 | S_mixture + v1_protocol.residual |

No se probó en esta etapa (por coste de cómputo, no por criterio): **A_surface** (features de cola de pista, presión salidas−llegadas,
minutos desde la salida anterior, congestión relativa; implementadas en `v2_features.py` y cubiertas por pruebas), **B** (pesos por
aeropuerto por LOFO; el análisis por régimen `tail_regime_weights_pooled.csv` mostró que la ganancia estaba en el régimen de la cola,
que S_mix resuelve de forma estructural) y **D** (ARIMA/SARIMAX: 12 meses de historia no permiten ajustar estacionalidad anual y la
señal horaria ya está capturada por las features de congestión). Se documentan como pendientes para una etapa 2.

## 2. El hallazgo que motiva S_mix

Cuando el aeropuerto no registra el off-block real, `BLOCK_TIME_UTC_mvt` se rellena con `SCHED_TIME_UTC_mvt` y el target vale
**exactamente** `MVT_TIME − SCHED_TIME` (±10 s). En 2025: 100 049 filas (4.8 %), 170 de los 202 extremos > 3 h, 1 506 de las 3 924 filas
entre 1 y 3 h. Una segunda familia (17 filas, LIRF sin plan de vuelo) vale ≈ 86 400 s + taxi (fecha del bloque desplazada un día).
Ninguno de los 202 extremos coincide con `MVT − AOBT_3`. Ambas familias son funciones conocidas de columnas presentes en ranking, así que
la predicción RMSE-óptima es la esperanza condicional: `p0·normal + p1·(MVT−SCHED) + p2·(86 400 + normal)`.

| y_bucket | n | pct_|r|<=10min | pct_|r|<=30min | median_r | r_p10 | r_p90 |
|---|---:|---:|---:|---:|---:|---:|
| 1-3h | 3023 | 0.1 | 0.2 | 3061.0 | 1024.0 | 4739.0 |
| <=1h | 2059538 | 0.9 | 1.0 | -52.0 | -363.0 | 306.0 |
| >3h | 16 | 0.1 | 0.2 | 10346.0 | -63.0 | 14401.0 |


| regime | n | pct_rows | p_extreme | mean_y | median_y | pct_of_SSE_vs_global_median |
|---|---:|---:|---:|---:|---:|---:|
| R1_normal | 2060294 | 98.813 | 0.0000 | 987 | 911 | 56.8 |
| R2_aobt3<2h_sched>6h | 2250 | 0.108 | 0.0000 | 1066 | 995 | 0.1 |
| R2b_aobt3<2h_lobt>3h | 14 | 0.001 | 0.6429 | 12029 | 11372 | 0.3 |
| R3_aobt3_2-6h | 18 | 0.001 | 0.2222 | 4854 | 1860 | 0.1 |
| R4_aobt3>6h | 1 | 0.000 | 1.0000 | 87002 | 87002 | 1.2 |
| R5_noFPL | 22262 | 1.068 | 0.0056 | 1253 | 949 | 13.4 |
| R6_noFPL_sched>6h | 208 | 0.010 | 0.2981 | 15491 | 1198 | 28.2 |

## 3. Cómo se evitó el leakage

- Cortes por `MVT_TIME_UTC_mvt`; cada fold entrena solo con meses anteriores al de validación; diciembre sellado hasta congelar la finalista.
- Las etiquetas de régimen se calculan **solo** con el target del bloque de entrenamiento; en validación y ranking nunca se leen.
- El clasificador y `direct_normal` usan exactamente las 29 features V1 (sin target, sin `BLOCK_TIME`); el residual V1 no se reentrena.
- `mvt_minus_sched` (rama clase 1) es `MVT_TIME − SCHED_TIME`, ambas presentes en ranking. Auditoría en `feature_audit.csv`.
- La validación se puntúa contra el target original completo (extremos incluidos); ninguna fila se excluye de la validación.

| variable | usada_en | usa_target | usa_BLOCK_TIME | disponible_en_ranking | estado |
|---|---|---|---|---|---|
| month | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| day_of_week | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| hour_utc | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| is_weekend | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| hour_sin | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| hour_cos | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| month_sin | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| month_cos | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| mvt_minus_aobt3 | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| mvt_minus_eobt1 | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| mvt_minus_lobt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| mvt_minus_iobt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| mvt_minus_sched | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| aobt3_minus_sched | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| has_aobt3 | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| departures_per_airport_hour | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| departures_per_airport_day | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| departures_per_airport_runway_hour | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| arrivals_per_airport_hour | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| ADEP_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| ADES_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| RUNWAY_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| STAND_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| AIRCRAFT_TYPE_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| MARKET_SEGMENT_flt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| FLIGHT_TYPE_flt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| WK_TBL_CAT_flt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| AIRCRAFT_OPERATOR_flt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| FLIGHT_RULE_mvt | S_mix (clasificador, direct_normal, residual_V1) | False | False | True | usada |
| regime_class (etiqueta) | SOLO como etiqueta de entrenamiento del clasificador y filtro de direct_normal | True | False | False | etiqueta (no feature) |
| mvt_minus_sched (en la mezcla) | valor de la rama clase 1: p1·(MVT − SCHED) | False | False | True | usada |
| dep_pm15 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| dep_pm30 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| rwy_prev10 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| rwy_prev15 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| rwy_prev30 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| arr_pm15 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| arr_pm30 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| net_pm15 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| net_pm30 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| min_since_prev_rwy_dep | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| min_to_next_rwy_sched | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| rel_dep_pm30 | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| rel_dep_hour | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |
| apt_hour_dow | A_surface (implementada y probada; NO entrenada en esta etapa por coste) | False | False | True | no probada |

## 4. Resultados pooled de desarrollo (mayo–noviembre; RMSE calculado sobre todas las filas, no media de medias)

| candidate | rmse | rmse_unweighted_fold_mean | mae | medae | cov_le300 | cov_le600 | cov_le900 | ae_p87 | ae_p90 | ae_p95 | ae_p99 | ae_max | rmse_normal | rmse_extreme | n_pred_negative | pct_pred_negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P | 390.6 | 379.7 | 161.1 | 111.8 | 87.71 | 97.76 | 99.16 | 292.0 | 329.4 | 438.5 | 832.8 | 86579.2 | 262.3 | 26659.0 | 103 | 0.008 |
| V1P_direct | 386.5 | 374.5 | 163.7 | 113.3 | 87.36 | 97.69 | 99.12 | 296.0 | 333.9 | 444.6 | 847.8 | 86434.9 | 284.7 | 24079.0 | 118 | 0.009 |
| V1P_residual | 434.6 | 422.0 | 162.7 | 112.2 | 87.60 | 97.73 | 99.16 | 293.4 | 331.0 | 441.0 | 835.2 | 87496.0 | 269.3 | 31431.3 | 143 | 0.011 |
| S_mix | 320.9 | 316.5 | 159.3 | 111.4 | 87.80 | 97.80 | 99.18 | 291.1 | 328.3 | 436.8 | 823.1 | 86653.0 | 259.4 | 17401.4 | 69 | 0.005 |
| S_mix_dn | 324.8 | 319.2 | 159.2 | 111.5 | 87.84 | 97.84 | 99.20 | 290.8 | 328.0 | 435.2 | 816.2 | 86596.2 | 259.0 | 18062.5 | 15 | 0.001 |
| S_mix_onV1P | 320.6 | 316.0 | 161.3 | 112.5 | 87.49 | 97.69 | 99.15 | 294.5 | 332.7 | 443.9 | 839.8 | 86571.7 | 268.8 | 16080.6 | 82 | 0.006 |
| S_mix_floor | 319.5 | 315.6 | 159.3 | 111.4 | 87.80 | 97.80 | 99.18 | 291.2 | 328.4 | 436.9 | 823.8 | 86653.0 | 259.6 | 17162.5 | 69 | 0.005 |

## 5. Por fold (V1P vs S_mix)

| candidate | fold | n | rmse | mae | medae | cov_le300 | cov_le600 | cov_le900 | n_extreme | rmse_extreme | rmse_normal | n_pred_negative |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P | May | 185202 | 324.7 | 152.2 | 108.5 | 88.72 | 98.13 | 99.40 | 17 | 23354.1 | 235.3 | 12 |
| V1P | Jun | 183142 | 403.7 | 161.1 | 111.6 | 87.80 | 97.77 | 99.16 | 18 | 30748.6 | 264.4 | 12 |
| V1P | Jul | 190713 | 559.0 | 182.7 | 119.1 | 85.36 | 96.69 | 98.57 | 50 | 28384.3 | 318.1 | 21 |
| V1P | Aug | 191182 | 357.2 | 163.7 | 114.0 | 87.33 | 97.63 | 99.11 | 25 | 20925.3 | 265.1 | 19 |
| V1P | Sep | 183950 | 394.2 | 160.9 | 113.0 | 87.54 | 97.92 | 99.28 | 21 | 27945.6 | 257.3 | 19 |
| V1P | Oct | 185674 | 253.9 | 154.5 | 109.9 | 88.58 | 98.07 | 99.28 | 14 | 6618.3 | 247.3 | 8 |
| V1P | Nov | 162332 | 365.5 | 150.8 | 106.7 | 88.89 | 98.17 | 99.37 | 6 | 46155.6 | 234.2 | 12 |
| S_mix | May | 185202 | 316.5 | 151.0 | 107.9 | 88.78 | 98.21 | 99.43 | 17 | 22461.2 | 232.0 | 10 |
| S_mix | Jun | 183142 | 344.6 | 159.3 | 111.0 | 87.91 | 97.82 | 99.19 | 18 | 22369.5 | 263.5 | 13 |
| S_mix | Jul | 190713 | 415.6 | 180.1 | 118.4 | 85.43 | 96.74 | 98.61 | 50 | 16484.6 | 318.5 | 22 |
| S_mix | Aug | 191182 | 293.2 | 161.5 | 113.2 | 87.42 | 97.69 | 99.12 | 25 | 11253.9 | 263.3 | 4 |
| S_mix | Sep | 183950 | 299.2 | 158.9 | 112.7 | 87.63 | 97.95 | 99.28 | 21 | 15082.9 | 252.0 | 15 |
| S_mix | Oct | 185674 | 246.4 | 153.4 | 109.8 | 88.69 | 98.10 | 99.30 | 14 | 4923.8 | 242.5 | 5 |
| S_mix | Nov | 162332 | 300.4 | 149.2 | 106.3 | 88.98 | 98.20 | 99.40 | 6 | 32327.3 | 227.1 | 0 |

## 6. Por aeropuerto (pooled de desarrollo)

| ADEP_mvt | V1P | S_mix | delta_rmse |
|---|---:|---:|---:|
| EDDF | 190.4 | 189.6 | -0.8 |
| EDDM | 182.2 | 180.6 | -1.6 |
| EGLL | 260.4 | 259.7 | -0.7 |
| EHAM | 174.5 | 173.0 | -1.5 |
| LEBL | 233.2 | 231.1 | -2.1 |
| LEMD | 197.2 | 196.1 | -1.1 |
| LFPG | 280.2 | 278.4 | -1.8 |
| LIRF | 1132.5 | 808.7 | -323.7 |
| LSZH | 353.8 | 353.0 | -0.9 |
| LTFM | 258.8 | 257.7 | -1.1 |


| ADEP_mvt | V1P | S_mix | delta_mae |
|---|---:|---:|---:|
| EDDF | 129.6 | 129.1 | -0.5 |
| EDDM | 120.4 | 120.0 | -0.4 |
| EGLL | 173.7 | 173.4 | -0.3 |
| EHAM | 121.3 | 120.3 | -1.0 |
| LEBL | 152.9 | 152.3 | -0.7 |
| LEMD | 137.7 | 137.4 | -0.3 |
| LFPG | 174.9 | 174.0 | -0.9 |
| LIRF | 305.8 | 290.1 | -15.7 |
| LSZH | 132.1 | 131.2 | -0.9 |
| LTFM | 179.1 | 178.3 | -0.8 |

## 7. Bootstrap pareado por bloques aeropuerto-día (1 000 remuestreos, SEED 42, vs V1P)

| candidate | rmse_baseline | rmse_candidate | delta_rmse_point | delta_p2.5 | delta_p50 | delta_p97.5 | p_improve | n_blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P_direct | 390.6 | 386.5 | -4.1 | -15.5 | -4.3 | 10.6 | 0.742 | 2140 |
| V1P_residual | 390.6 | 434.6 | 44.0 | 34.1 | 44.0 | 52.6 | 0.000 | 2140 |
| S_mix | 390.6 | 320.9 | -69.7 | -93.7 | -68.5 | -42.6 | 1.000 | 2140 |
| S_mix_dn | 390.6 | 324.8 | -65.7 | -91.3 | -64.7 | -35.9 | 1.000 | 2140 |
| S_mix_onV1P | 390.6 | 320.6 | -70.0 | -95.0 | -68.7 | -40.8 | 1.000 | 2140 |
| S_mix_floor | 390.6 | 319.5 | -71.0 | -95.4 | -69.6 | -43.9 | 1.000 | 2140 |

## 8. Criterios de selección (solo desarrollo)

| candidate | rmse_gain_s | cov5_gain_pp | folds_not_worse | worst_fold_worsening_pct | min_cov5_fold | rmse_normal_gain_s | share_gain_top50_rows | p_improve_boot | all_criteria |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| V1P_direct | 4.1 | -0.35 | 4/7 | 3.2 | 84.8 | -22.3 | 6.100 | 0.742 | False |
| V1P_residual | -44.0 | -0.11 | 0/7 | 15.5 | 85.3 | -7.0 | nan | 0.000 | False |
| S_mix | 69.7 | 0.09 | 7/7 | -2.5 | 85.4 | 2.9 | 0.939 | 1.000 | False |
| S_mix_dn | 65.7 | 0.12 | 7/7 | -2.5 | 85.4 | 3.3 | 0.988 | 1.000 | False |
| S_mix_onV1P | 70.0 | -0.22 | 7/7 | -1.5 | 85.1 | -6.5 | 1.020 | 1.000 | False |
| S_mix_floor | 71.0 | 0.08 | 7/7 | -2.5 | 85.4 | 2.7 | 0.942 | 1.000 | False |

Lectura del criterio 8 ("no depender de pocas filas"): el 94 % de la reducción de SSE de S_mix procede de las 50 filas con mayor
ganancia, **por construcción**: la mejora estructural está en el régimen de bloque = hora programada, que por definición son pocas filas
con errores de miles de segundos. No es un artefacto de un fold ni de un aeropuerto: S_mix mejora el RMSE en 7/7 folds (de −7.6 a −143 s),
en los 10 aeropuertos, el MAE en los 10 aeropuertos, el RMSE de la población normal (0 < y ≤ 3 h) en 2.9 s y reduce las predicciones
negativas (103 → 69). Se acepta el candidato con esa salvedad explícita.

## 9. Intervalos conformales (diagnóstico; calibrados con los folds anteriores, evaluados en el siguiente)

| candidate | nominal | cobertura_media | cobertura_min | semiancho_medio_s |
|---|---:|---:|---:|---:|
| S_mix | 0.9 | 0.8682 | 0.8392 | 291 |
| S_mix | 0.9 | 0.8985 | 0.8722 | 328 |
| S_mix_dn | 0.9 | 0.8682 | 0.8378 | 291 |
| S_mix_dn | 0.9 | 0.8985 | 0.8712 | 328 |
| S_mix_floor | 0.9 | 0.8682 | 0.8392 | 291 |
| S_mix_floor | 0.9 | 0.8985 | 0.8722 | 328 |
| S_mix_onV1P | 0.9 | 0.8680 | 0.8391 | 294 |
| S_mix_onV1P | 0.9 | 0.8984 | 0.8726 | 332 |
| V1P | 0.9 | 0.8681 | 0.8396 | 292 |
| V1P | 0.9 | 0.8984 | 0.8718 | 329 |

No se incluyen intervalos en la submission (el concurso pide predicción puntual).

## 10. Holdout de diciembre (abierto una sola vez, con la finalista ya congelada)

| candidate | rmse | mae | medae | cov_le300 | cov_le600 | cov_le900 | n_extreme | rmse_extreme | rmse_normal | n_pred_negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1P | 332.2 | 152.4 | 107.6 | 88.82 | 98.12 | 99.33 | 12 | 27255.2 | 237.7 | 17 |
| V1P_direct | 336.2 | 155.1 | 109.2 | 88.46 | 98.03 | 99.29 | 12 | 26463.8 | 249.6 | 27 |
| V1P_residual | 375.7 | 153.9 | 108.2 | 88.76 | 98.08 | 99.30 | 12 | 32830.1 | 251.1 | 17 |
| S_mix | 241.1 | 150.6 | 107.6 | 88.86 | 98.15 | 99.34 | 12 | 6981.5 | 233.6 | 2 |
| S_mix_dn | 241.0 | 151.3 | 108.3 | 88.76 | 98.16 | 99.34 | 12 | 7014.5 | 233.5 | 1 |
| S_mix_onV1P | 242.6 | 151.9 | 108.4 | 88.62 | 98.10 | 99.32 | 12 | 6256.3 | 236.6 | 7 |
| S_mix_floor | 241.3 | 150.6 | 107.6 | 88.86 | 98.14 | 99.33 | 12 | 6981.5 | 233.8 | 2 |

## 11. Decisión

**CREAR V2 (S_mix) — sustituye a V1 como recomendación; V1 se conserva intacta como baseline** — Desarrollo pooled (may–nov): RMSE 390.6 → 320.9 (−69.7 s), 7/7 folds y 10/10 aeropuertos mejoran, MAE 161.1 → 159.3, cobertura ≤5 min 87.71 → 87.80 %, bootstrap P(mejora)=1.000. Holdout diciembre (abierto una vez): RMSE 332.2 → 241.1 (−91.0 s), MAE 152.4 → 150.6, cobertura ≤5 min 88.82 → 88.86 %, negativos 17 → 2. Único criterio no superado formalmente: c8 (la ganancia se concentra en las filas del régimen bloque=hora programada), por construcción; se acepta con esa salvedad porque la mejora es consistente en todos los folds y aeropuertos y también en la población normal.
