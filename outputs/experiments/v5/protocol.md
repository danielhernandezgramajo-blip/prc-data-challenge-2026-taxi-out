# V5 — Rutas nuevas (exploración). Protocolo escrito antes de mirar enero/julio

Fecha: 2026-10-02. Base: V4b C2 (rama normal V4b + compuerta multivariable, variante S). Reglas de siempre: los 14 Parquet no se
tocan; sin leaderboard para elegir nada; sin subidas ni push; MVT_ID solo para alinear.

## 1. Cribado (meses internos febrero + agosto 2025; enero/julio no se miran)

Mismos 9 meses de entrenamiento que los componentes de V4b de cada fold (LOJO: mar–jul + sep–dic; LOJuO: ene + mar–jun +
sep–dic), misma compuerta (c1_clf) y misma celda de fecha desplazada (c0_clf) de V4b; solo cambia la rama normal.

- E1 `linear_tree`: LightGBM (2000 it, lr 0.05, 255 hojas, sin early stopping) con hojas constantes y con hojas lineales;
  categóricas de cardinalidad alta con media del target fuera de mes (k = 20). Comparado con c2_n (CatBoost V4b).
- E2 especialista sin registro NM: auditoría del grupo A (sin AOBT/EOBT/LOBT/IOBT) y tabla por banda de MVT − SCHED frente a
  la compuerta V4b; cota de oráculo.
- Diagnósticos de contexto (sin target del ranking): taxi-in de llegadas contemporáneas, taxi NM de salidas vecinas,
  descomposición del residuo por aeropuerto-día y aeropuerto-día-hora, AOBT_3 frente a BLOCK, tasas de artefacto ARR 2025 vs 2026.

## 2. Comprobación única en enero/julio (outer), fijada antes de verla

Solo para la mezcla de ramas normales `b = 0.5·c2_n + 0.5·lgb_const` (semilla 42 de LightGBM con cada semilla de c2_n),
combinada con la compuerta y la celda de V4b (variante S) y promediada sobre las 3 semillas, frente a C2 (variante S) con las
mismas semillas. Métrica: RMSE_w (pesos 0.4429/0.5571) y bootstrap pareado por bloques aeropuerto-día (1000 réplicas).
Regla: se propone como candidata V5a solo si ΔRMSE_w ≤ −1 s y mejora en los dos meses; si no, se descarta.

## 3. Ruta externa (decisión del usuario)

OPDI (EUROCONTROL PRU + OpenSky, abierto): sondeo en dos ventanas de 10 días de febrero y agosto 2025 (`src/v5_opdi_probe.py`,
se ejecuta en el Mac). Criterio de continuación fijado antes de verlo: algún evento en tierra con cobertura ≥ 50 % de las
salidas de clase normal y |TAXITIME − (MVT − evento) − mediana| ≤ 60 s en ≥ 50 % de esas filas (el AOBT_3 del NM llega a
≈ 20 %). Si no se cumple, la ruta OPDI se descarta.

## 2b. Compuerta mezclada (añadido 2026-10-02 antes de ver ningún resultado de la compuerta LightGBM)

Compuerta LightGBM (`src/v5_gate.py`, 1000 it, mismas features y submuestreo que c1_clf). Cribado en feb/ago con
`p1 = 0.5·c1_clf + 0.5·lgb_gate`. Solo si baja el RMSE total interno en los dos folds se hace UNA comprobación outer de
"rama normal mezclada + compuerta mezclada" frente a "rama normal mezclada", con la misma regla (ΔRMSE_w ≤ −1 s y mejora en
enero y julio).

## Resultado §2 (rama normal mezclada, una sola mirada a ene/jul)

ΔRMSE_w = −1.41 s (IC 95 % bootstrap [−2.01, −0.87]); enero −0.80 s, julio −1.94 s → cumple la regla. Ganancia pequeña pero
estable (en feb/ago: −1.5 y −2.0 s en total; −2.5 y −2.6 s sin la anomalía de LTFM).

## Resultado §3 (OPDI, 2026-10-02) → RUTA DESCARTADA

Ventana 4–13 feb 2025 (descargada en el Mac de Daniel; el sondeo de agosto no llegó a ejecutarse porque el script fallaba con
el esquema real —columnas en minúscula y eventos enlazados por `flight_id`—, ya corregido). 95 % de las salidas DEP se
emparejan con un vuelo OPDI (ADEP + callsign + ventana temporal), pero los eventos en tierra del aeropuerto de salida casi no
existen: `exit-parking_position` cubre el 5.8 % de las salidas de clase normal ≤ 1 h (sobre todo LSZH y LEBL) y `exit-apron`
el 4 %; `entry-taxiway` (primero) el 17 % con solo el 10 % a ±60 s; `entry-runway` (60 %) y `first_seen` (95 %) no miden el
off-block (10–13 % a ±60 s). El criterio previo (cobertura ≥ 50 % y ±60 s en ≥ 50 %) no se cumple. Además, donde existen, los
proxies OPDI predicen peor que la rama normal actual (p. ej. LSZH exit-parking: RMSE 257 s frente a 151 s de c2_n; mezclarlos
50/50 también empeora). Agosto no puede cambiar una cobertura del 6 % al 50 %, así que no se repite.

## Diagnósticos finales (2026-10-02, sin target del ranking)

- ARVT_1/ARVT_3 (ablación pendiente de V4b): explican ≤ 0.13 % de la varianza del residuo de la clase normal → descartado.
- MVT_ID no codifica el orden de BLOCK (Spearman con MVT, SCHED y BLOCK idéntico, ≈ 0.14 por aeropuerto-día) → sin fuga.
- Compuerta calibrada en filas de alto riesgo (|s − b| > 1 h): p1 medio ≈ tasa real de clase 1 por quintiles.
- RMSE oficial esperado de V4b si está calibrado: ≈ 294 s; la incertidumbre por azar en las filas de cola es grande
  (±1σ ≈ [265, 319]); el 92 % de esa varianza viene de las 72 filas de la celda de fecha desplazada (25 en LIRF).
