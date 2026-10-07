# V7 — Corrección por congestión extrema (protocolo, 2026-10-06; fijado antes de calcular la validación)

Base: V6 (V5a + regla LOBT). Reglas de siempre: 14 Parquet intactos; sin subidas ni push; MVT_ID solo para alinear; el
leaderboard no se usa para elegir ni ajustar nada (los scores de V3/V4b/V6 solo se registran).

## Motivación (solo features del ranking + targets de 2025)

- "Congestión de vecinos" nb60 = media de MVT − AOBT_3 (si está en (0, 2 h)) de las demás salidas del mismo aeropuerto en
  ±60 min (sin la propia fila); anomalía nba = nb60 − mediana 2025 del aeropuerto. Solo usa AOBT_3 de otros vuelos, que
  está en el fichero del ranking (como las features de congestión que ya usamos); ningún target.
- El ranking 2026 tiene muchísima más congestión extrema que 2025: filas con nba > 1800 s: 647 (ene+jul 2025: 13);
  > 2400 s: 138 (ene+jul 2025: 0). Casi todo EHAM 3–9 ene 2026, LFPG 5–7 ene 2026, LTFM 19 ene 2026.
- En la validación 2025 (meses OOF) el modelo se queda corto cuando hay congestión: banda nba 900–1800 → y medio 2583 s
  frente a predicción 1965 s; ahí predecir y ≈ MVT − AOBT_3 propio (A) tiene menos error que el modelo (RMSE 1146 vs 1227).
  Es el mismo fallo de extrapolación de los árboles que corrigió la regla LOBT, ahora en el tiempo de rodaje normal.

## Corrección (una configuración principal)

- Filas: fuera de la celda de fecha desplazada, MVT − LOBT ≤ 2 h o nulo (si no, manda la regla LOBT), A = MVT − AOBT_3 en
  (0, 7200) s, nba > 900 s.
- Sobre la rama normal b (antes de mezclar con la compuerta): b' = b + c + w·(A − b), con (c, w) por mínimos cuadrados en
  las filas de clase 0 (y ≤ 3 h) de esa banda en los meses de ajuste. Después se recombina igual que V6.
- Validación: para el outer de cada fold (enero / julio), (c, w) se ajustan con las predicciones inner (feb + ago) del MISMO
  fold, que son fuera de muestra. Sensibilidad: umbral 600 y 1200 s (se informan, no se eligen).
- Regla de adopción (fijada ahora): mejora del RMSE total en enero Y en julio, mejora del RMSE de las filas afectadas en
  enero y en julio, y IC 95 % bootstrap (aeropuerto-día) de ΔRMSE_w con extremo superior < 0. Si no se cumple, se descarta.
- Final: (c, w) con los 4 meses OOF (ene, feb, jul, ago), aplicados a la b final del ranking.

### Enmienda (antes de ver ningún resultado; el primer intento falló por formato al imprimir)

Julio 2025 casi no tiene filas con nba > 900 (el outer de LOJuO puede quedar con 0 filas afectadas), así que "mejora en
julio" no es evaluable. Comprobaciones que cuentan:
1. LOJO: ajuste con feb + ago (inner) → evaluación en enero (outer).
2. LOJO cruzado: ajuste con enero → evaluación en febrero (ambos fuera de muestra para los modelos LOJO).
3. LOJuO: ajuste con feb + ago → evaluación en julio; cuenta solo si hay ≥ 30 filas afectadas (si no, neutra).
Adopción: en cada comprobación que cuenta, mejora del RMSE total y del RMSE de las filas afectadas, y extremo superior del
IC 95 % bootstrap de ΔRMSE_w (ene/jul) < 0.

## Resultado (2026-10-06) → DESCARTADA

| umbral nba | ΔRMSE_w ene/jul (IC 95 %) | ene (ajuste feb+ago) | jul | ene → feb |
|---|---|---|---|---|
| 900 (principal) | +0.11 s [−0.29, +0.55] | +0.21 s (afectadas 637→660) | 2 filas | −3.30 s (981→871) |
| 600 | +0.09 s [−0.53, +0.63] | +0.31 s | −0.11 s | −2.59 s |
| 1200 | +0.13 s [+0.01, +0.30] | +0.27 s | 0 filas | +1.25 s |

El peso ajustado sobre A − b cambia de signo según el mes (w = −0.30 con feb+ago, +0.06 con enero): en 2025 el AOBT_3
propio no es fiable en congestión (muchos casos de AOBT viejo). No cumple la regla → V7 no se construye.
Queda constancia de que el ranking 2026 tiene episodios de congestión fuera del rango de 2025 (EHAM 3–9 ene, LFPG 5–7 ene,
LTFM 19 ene) donde el modelo probablemente se queda corto, pero 2025 no permite calibrar una corrección.
(`outputs/experiments/v7/congestion_check.json`)

## V7b — corrección anclada en la congestión de los vecinos (prerregistrada 2026-10-07, antes de calcularla)

Motivo (visto en 2025 antes de este registro): en los meses OOF, con anomalía de vecinos alta, la media de y queda ≈ 300 s
por ENCIMA de la media de MVT − AOBT_3 de los vecinos (nb) y el modelo muy por debajo (banda 900–1200: y 2432, nb 2108,
predicción 1965; banda 1200–1800: y 2861, nb 2561, predicción 1990). nb promedia muchos vuelos, así que un AOBT_3 viejo
de un vuelo concreto pesa poco (a diferencia del A propio de V7, que falló por eso).

- Principal: filas con nba > 600 s, fuera de la celda y sin regla LOBT (no exige A propio): b' = b + c + w·(nb − b),
  (c, w) por mínimos cuadrados en clase 0 (y ≤ 3 h) de los meses de ajuste.
- Se informan también (no se eligen): umbral 300 y 900; variante "solo sesgo" (b' = b + c).
- Mismas comprobaciones y misma regla de adopción que la enmienda de V7 (feb+ago → ene; ene → feb; jul si ≥ 30 filas;
  mejora total y en filas afectadas en todas las que cuentan; IC 95 % de ΔRMSE_w con extremo superior < 0).

### Resultado V7b (2026-10-07) → DESCARTADA
nba > 600: ΔRMSE_w +0.09 s (IC [−0.75, +0.92]); feb+ago→ene +0.14 s, →jul +0.05 s, ene→feb −5.79 s. Mismo patrón con 300/900
y con "solo sesgo": la corrección aprendida en febrero (episodio LTFM) se pasa en enero. El peso sobre nb − b sale ≈ 0.
(`outputs/experiments/v7/neighbour_check.json`)

## V7c — solo filas SIN AOBT_3 propio (prerregistrada 2026-10-07, antes de calcularla)

Motivo (visto en 2025 tras V7b; tercera variante, se dice explícitamente): el modelo usa el AOBT_3 propio como señal
principal; las filas que no lo tienen (en el ranking, todas sin plan NM) solo ven features estáticas y conteos de
congestión. En los meses OOF de 2025, con congestión (nba > 600), esas filas quedan muy por debajo en todos los meses con
filas: enero 600–900 y 1933 vs predicción 1151 (nb 1705); enero > 900 y 2409 vs 1189 (nb 2102); febrero > 900 y 4844 vs
1645 (nb 2457). En el ranking hay 489 filas así con nba > 900 (315 en EHAM 3–5 ene 2026): V6 media 1499 frente a nb 2591.

- Principal (sin parámetros, nada que ajustar): en filas con has_aobt3 = 0, fuera de la celda, sin regla LOBT y nba > 600:
  b' = max(b, nb). Sensibilidad: umbral 300 y 900; versión ajustada b' = b + c + w·(nb − b).
- Como no se ajusta nada, enero, febrero y julio son prueba directa. Adopción: mejora del RMSE total y del de las filas
  afectadas en cada mes con ≥ 30 filas afectadas, y IC 95 % bootstrap de ΔRMSE_w (ene/jul) con extremo superior < 0.

### Resultado V7c (2026-10-07) → CUMPLE la regla (principal, sin parámetros)
max(b, nb), nba > 600: ΔRMSE_w −0.35 s (IC 95 % [−0.86, −0.01]); enero 112 filas −0.74 s (RMSE afectadas 1372→1084),
febrero 173 filas −8.84 s (3290→2595); julio 17 y agosto 3 filas (neutras, < 30). Sensibilidad: umbral 300 no pasa
(jul/ago +0.15/+0.10), 900 casi (IC hasta +0.02); la versión ajustada (c, w) empeora (sobreajusta febrero) → se queda la
principal. Es la tercera variante probada sobre los mismos meses: la elección no es ciega, pero la principal no tiene
parámetros que ajustar y la dirección es la misma en todos los meses con congestión. (`outputs/experiments/v7/noaobt_check.json`)
