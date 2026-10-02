# V4b — Resultado mínimo C0/C1/C2

Generado por `src/v4b_report.py`. Protocolo: `protocol.md` (enmienda A1 de cómputo incluida). Validación: LOJO (enero 2025) y
LOJuO (julio 2025), componentes entrenados con 9 meses, calibración anidada en febrero/agosto, ensamble de semillas
42, 7, 2025. Sin candidata, sin subidas, sin push, sin leaderboard. Las cotas de oráculo de V4 (230.2, 205.7)
no son alcanzables y no se usan como referencia.

## Veredicto

**No pasa** el umbral de −15 s: C2 − C0 = **-14.6 s** de RMSE ponderado (IC 95 % [-25.7, -0.8],
semillas -14.2, -13.7, -12.2). Compuerta (C1 − C0): -10.0 s; rama normal (C2 − C1):
-4.6 s. Según la regla prefijada, no se continúa con modelos costosos.

## Lectura

- **¿Rama normal insuficiente? Sí.** Sin compuerta, el RMSE de clase 0 baja -9.4 s en enero y -19.2 s en julio (≤ 1 h: -8.8 / -12.5 s), igual en las 3 semillas; C2 mejora a C0 en 10 de 10 aeropuertos.
- **¿Compuerta mal calibrada? Solo en parte.** El clasificador nuevo mejora poco la discriminación (AUC 0.869 → 0.876 en enero, 0.879 → 0.887 en julio); su p medio en filas de clase 1 sigue en ≈ 0.21. La ganancia de C1 se concentra en julio (-19.8 s) y en LIRF (-63.5 s); en enero C1 empeora (+1.3 s). En la selección anidada ganó la mezcla suave simple: la recalibración por SSE, la compuerta dura y las regresiones de corrección quedaron peor en el SSE interno cruzado.
- **Una sola fila oculta parte de la mejora de la rama normal.** LIRF (jul), clase 1, y = MVT − SCHED = 87 002.0 s: la mezcla con la normal de V3 quedaba más cerca por casualidad (la normal de V3 extrapolaba un valor enorme para esa fila). Esa fila sube el SSE de C2 frente a C1 en 1.78e+09; sin ella, C2 − C1 pasaría de -4.6 a -12.5 s. El RMSE de clase 1 de julio (C1 719.9 → C2 847.7) empeora por esa fila, no por un sesgo sistemático.
- **La decisión frente al umbral no es robusta.** Quitar una sola fila cambia C2 − C0 de -14.6 s a entre -20.7 y -14.0 s (las 6 filas más influyentes). La mejora frente a V3 es real (IC 95 % por debajo de 0, 3 de 3 semillas, 10 de 10 aeropuertos); si llega o no a 15 s no se puede decidir con estos datos. Se aplica la regla tal como se fijó.
- **Fecha desplazada aparte:** sin efecto medible (misma celda en C0–C2; 1 fila en enero y 4 en julio).
- **Ruido:** en enero, 3 filas de clase 0 con y > 3 h dominan el SSE; sin ellas (sensibilidad que no decide) C2 − C0 sería -16.6 s.
- **Selección honesta:** variantes no elegidas parecen mejores en el mes outer (la mejor, suave + recalibración por SSE con término de tamaño de Δ · p1 nuevo · normal V4b: -24.3 s), pero quedaron peor en febrero/agosto. Elegirlas ahora sería ajustar con el mes de evaluación. Posible sesgo del esquema: el ajuste cruzado con un solo mes interno penaliza más a las calibraciones flexibles; un protocolo nuevo podría usar más meses internos.
- **No ejecutado (regla de −15 s no superada):** ablación con/sin ARVT, adaptación no supervisada al ranking, max_ctr_complexity 3, ensamble con LightGBM.

## 1. RMSE ponderado (ensamble de semillas; enero y julio 2025 directos)

| config | definición (elegida en feb/ago) | enero | julio | RMSE_w | rango semillas | Δ vs C0 |
|---|---|---|---|---|---|---|
| **C0** | C0 · V3 reentrenado | 359.0 | 342.9 | **350.1** | 349.9–351.9 | — |
| **C1** | C1 · suave + Platt (log-loss) · p1 nuevo · normal V3 | 360.3 | 323.2 | **340.1** | 339.9–343.3 | -10.0 |
| **C2** | C2 · suave + Platt (log-loss) · p1 nuevo · normal V4b | 351.6 | 322.2 | **335.5** | 335.7–338.1 | -14.6 |

## 2. Diferencias con bootstrap pareado (aeropuerto-día, estratificado por mes, 1 000 réplicas)

| comparación | Δ RMSE_w | IC 95 % | P(mejora) | por semilla | Δ enero | Δ julio |
|---|---|---|---|---|---|---|
| C1-C0 | -10.0 | [-17.2, -4.4] | 1.00 | -8.6, -8.6, -10.0 | +1.3 | -19.8 |
| C2-C0 | -14.6 | [-25.7, -0.8] | 0.98 | -14.2, -13.7, -12.2 | -7.4 | -20.7 |
| C2-C1 | -4.6 | [-14.9, +12.1] | 0.72 | -5.6, -5.2, -2.2 | -8.7 | -0.9 |

Sensibilidad (no decide): sin las filas de clase 0 con y > 3 h (3 en enero, 3 en julio), C1 − C0 = -11.6 s,
C2 − C0 = -16.6 s, C2 − C1 = -5.0 s (RMSE_w de C0 303.4).
Ninguna configuración predice esas filas; pesan mucho en el SSE y comprimen las diferencias medidas en RMSE.

## 3. Desglose por clase (ensamble; s)

| config | mes | total | clase 0 | clase 0 ≤ 1 h | clase 0 1–3 h | clase 0 > 3 h | clase 1 | clase 2 (n) | SSE % c0/c1/c2 |
|---|---|---|---|---|---|---|---|---|---|
| C0 | ene | 359.0 | 356.7 | 219.5 | 2 162.8 | 58 754.7 | 326.3 | 20 269.7 (1) | 94/4/2 |
| C0 | jul | 342.9 | 292.2 | 264.8 | 2 649.3 | 8 790.6 | 826.4 | 18 453.4 (4) | 70/24/6 |
| C1 | ene | 360.3 | 357.0 | 220.2 | 2 155.9 | 58 755.2 | 347.5 | 20 269.7 (1) | 93/5/2 |
| C1 | jul | 323.2 | 280.8 | 252.0 | 2 658.5 | 8 776.9 | 719.9 | 18 453.4 (4) | 72/21/7 |
| C2 | ene | 351.6 | 347.8 | 211.8 | 1 771.1 | 58 679.7 | 346.2 | 20 298.2 (1) | 93/5/2 |
| C2 | jul | 322.2 | 263.6 | 240.8 | 2 284.4 | 8 120.8 | 847.7 | 18 442.2 (4) | 64/29/7 |

## 4. ¿Rama normal insuficiente? RMSE en filas de clase 0, sin compuerta (ensamble)

| mes | rama normal | clase 0 | clase 0 ≤ 1 h |
|---|---|---|---|
| ene | blend 50/50 | 345.4 | 204.8 |
| ene | normal_C0 (V3) | 351.9 | 211.4 |
| ene | normal_C2 (V4b) | 342.5 | 202.6 |
| jul | blend 50/50 | 254.4 | 226.6 |
| jul | normal_C0 (V3) | 267.6 | 235.7 |
| jul | normal_C2 (V4b) | 248.4 | 223.2 |

Ablación ctr 2 (rama normal): **no ejecutada** — el ajuste superó dos veces el límite de memoria del entorno (~5.8 GiB); su efecto queda sin medir (protocolo §10, A2).

## 5. ¿Compuerta mal calibrada? Clasificadores de clase 1 (filas fuera de la celda; media de semillas)

| mes | clasificador | AUC | log-loss | Brier ponderado Δ² | p medio clase 1 | p medio clase 0 |
|---|---|---|---|---|---|---|
| ene | p1_C0 (V3) | 0.869 | 0.1458 | 0.0013 | 0.197 | 0.0433 |
| ene | p1_new (V4b) | 0.876 | 0.1432 | 0.0016 | 0.215 | 0.0426 |
| jul | p1_C0 (V3) | 0.879 | 0.1272 | 0.0057 | 0.184 | 0.0326 |
| jul | p1_new (V4b) | 0.887 | 0.1231 | 0.0042 | 0.214 | 0.0324 |

Variantes de etapa 2 por SSE interno cruzado (feb↔ago, sumado sobre outer × semillas). El RMSE_w outer se muestra solo como
diagnóstico: no se usó para elegir.

| config | rango | variante · fuente · normal | SSE interno (×1e9) | RMSE_w outer | Δ vs C0 |
|---|---|---|---|---|---|
| C1 | 1 | suave + Platt (log-loss) · p1 nuevo · normal V3 | 153.368 | 340.1 | -10.0 |
| C1 | 2 | mezcla suave (p tal cual) · p1 nuevo · normal V3 | 153.405 | 340.1 | -10.1 |
| C1 | 3 | suave + recalibración por SSE con término de tamaño de Δ · p1 nuevo · normal V3 | 157.133 | 337.9 | -12.2 |
| C1 | 4 | suave + recalibración por SSE · p1 nuevo · normal V3 | 157.501 | 339.2 | -10.9 |
| C1 | 5 | corrección estructurada b + q·Δ · p1 nuevo · normal V3 | 161.622 | 336.4 | -13.7 |
| C1 | 6 | compuerta dura · p1 nuevo · normal V3 | 161.661 | 344.8 | -5.3 |
| C1 | 7 | suave + recalibración por SSE · p1 de V3 · normal V3 | 162.102 | 351.7 | +1.6 |
| C1 | 8 | suave + recalibración por SSE con término de tamaño de Δ · p1 de V3 · normal V3 | 162.832 | 348.4 | -1.7 |
| C2 | 1 | suave + Platt (log-loss) · p1 nuevo · normal V4b | 140.301 | 335.5 | -14.6 |
| C2 | 2 | mezcla suave (p tal cual) · p1 nuevo · normal V4b | 140.340 | 335.3 | -14.8 |
| C2 | 3 | suave + recalibración por SSE con término de tamaño de Δ · p1 nuevo · normal V4b | 144.081 | 325.8 | -24.3 |
| C2 | 4 | suave + recalibración por SSE · p1 nuevo · normal V4b | 144.451 | 332.9 | -17.2 |
| C2 | 5 | suave + Platt (log-loss) · p1 nuevo · 50/50 V3 + V4b | 144.505 | 334.9 | -15.3 |
| C2 | 6 | mezcla suave (p tal cual) · p1 nuevo · 50/50 V3 + V4b | 144.540 | 334.7 | -15.4 |
| C2 | 7 | suave + recalibración por SSE con término de tamaño de Δ · p1 nuevo · 50/50 V3 + V4b | 148.115 | 329.5 | -20.7 |
| C2 | 8 | compuerta dura · p1 nuevo · normal V4b | 148.260 | 334.9 | -15.3 |

## 6. Aeropuertos (RMSE ponderado ene/jul de cada aeropuerto)

| aeropuerto | C0 | Δ C1 | Δ C2 |
|---|---|---|---|
| EDDF | 198.0 | -0.0 | -8.9 |
| EDDM | 194.4 | -0.1 | -3.5 |
| EGLL | 304.6 | -0.5 | -36.3 |
| EHAM | 207.5 | -0.0 | -9.8 |
| LEBL | 230.9 | -0.1 | -7.6 |
| LEMD | 192.1 | -0.2 | -9.9 |
| LFPG | 578.6 | -0.1 | -10.1 |
| LIRF | 701.3 | -63.5 | -27.9 |
| LSZH | 219.1 | -0.1 | -13.6 |
| LTFM | 279.7 | -0.2 | -15.7 |

C2 mejora a C0 en 10 de 10 aeropuertos.

## 7. Cómputo

| componente | ajustes | min por ajuste |
|---|---|---|
| c0_clf | 6 | 3.5 |
| c0_dn | 6 | 5.5 |
| c0_rn | 6 | 5.4 |
| c1_clf | 6 | 11.7 |
| c2_n | 6 | 39.0 |
