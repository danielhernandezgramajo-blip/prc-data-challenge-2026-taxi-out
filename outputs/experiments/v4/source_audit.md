# V4 · Etapa A — Auditoría de las fuentes visibles de BLOCK_TIME

Generado por `src/v4_report.py` a partir de las salidas de `src/v4_audit.py` (CSV/JSON de esta carpeta) y `src/v4_oracles.py`.
Datos: 190 713 salidas de julio y 165 677 de diciembre para los oráculos; 2 085 047 salidas (DEP)
de training 2025 para la auditoría. Protocolo prerregistrado: `protocol.md` (guardado antes de esta auditoría).

**Resultado: Gate A NO superado.** A1 pasa (masa de copia no-SCHED 5.77 % ≥ 5 %), A2 falla
(oráculo -5.25 s, IC 95 % [-5.8, -4.7] s, frente a −15 s exigidos).

## 1. Definiciones

- **Ancla**: timestamp visible en ranking del que BLOCK podría haberse copiado: SCHED_TIME, IOBT, LOBT, EOBT_1 y AOBT_3, cada
  uno con desplazamiento 0, −1 día y +1 día (15 anclas). `d = BLOCK − ancla`.
- **Atribución por fila**: ancla más cercana si |d| ≤ τ (τ = 30 s prefijado; 10 s = copia estricta). Empates exactos por
  precedencia SCHED > IOBT > LOBT > EOBT_1 > AOBT_3 (no altera la predicción: las anclas coinciden). Ambigua = ≥ 2 anclas
  dentro de τ con timestamps que difieren > 10 s.
- **Masa de copia**: exceso de filas con |d| ≤ 10 s sobre el fondo local (densidad por segundo en 30 < |d| ≤ 120 s), en % de
  todas las DEP. Separa copias de simples cercanías entre dos medidas del mismo evento. *Corregida*: d centrado con la moda por
  (aeropuerto, ancla) — la "corrección residual" de la hipótesis.

## 2. Cobertura por fuente (% de todas las DEP 2025)

| fuente | disponible | ±10 s | ±30 s | ±60 s | ±300 s | masa de copia | corregida | mediana d (|d|≤120 s) |
|---|---|---|---|---|---|---|---|---|
| SCHED_TIME | 100.00 | 4.80 | 5.44 | 9.57 | 36.62 | 3.44 | 3.39 | +1 |
| IOBT | 98.92 | 5.40 | 6.20 | 11.18 | 43.80 | 3.77 | 3.77 | +1 |
| LOBT | 98.92 | 5.38 | 6.18 | 11.14 | 43.65 | 3.75 | 3.76 | +1 |
| EOBT_1 | 98.92 | 5.83 | 6.70 | 12.52 | 48.42 | 3.93 | 4.05 | +3 |
| AOBT_3 | 98.92 | 8.25 | 10.44 | 20.79 | 73.84 | 4.92 | 5.23 | +3 |
| unión (15 anclas) | 100.00 | 14.85 | 17.96 | 32.74 | 87.25 | — | — | — |
| unión sin SCHED | 100.00 | 14.48 | 17.58 | 32.18 | 86.13 | — | — | — |

- Las 10 anclas desplazadas ±1 día no explican nada: la mayor cubre 0.0005 % de las DEP a ±300 s. La clase de fecha
  desplazada de V3 (y ≈ 86 400 s + rodaje, 13 filas en 2025) no es una copia de una fuente visible desplazada; es BLOCK con la
  fecha equivocada (experto `none − 86 400` del protocolo).
- La "corrección residual" apenas cambia nada: la suma de masas no-SCHED pasa de 16.37 % a
  16.81 % (la mediana de d entre las filas con |d| ≤ 120 s está a +1/+3 s según la fuente).
- La suma de masas por ancla cuenta varias veces la misma fila (IOBT/LOBT/EOBT_1 suelen coincidir con SCHED). Sin doble
  conteo, la masa de copia de las anclas no-SCHED en filas que no son copia de SCHED es **5.77 %**.

## 3. Atribución por fila

- τ = 30 s: **17.96 %** de las DEP se atribuyen a alguna ancla; τ = 10 s: 14.85 %.
  Ambiguas: 0.021 %. Atribuidas a anclas desplazadas: 0.00014 %.
- Ancla más cercana (sin tolerancia): AOBT_3 59.1 %, SCHED 26.2 %, EOBT_1 8.0 %,
  IOBT 6.6 %, LOBT 0.05 % (LOBT casi siempre coincide con IOBT y pierde por precedencia).
  Distancia a la más cercana: p25 56 s, mediana 116 s, p75 192 s.
- Fuentes ausentes: IOBT, LOBT, EOBT_1 y AOBT_3 faltan juntas en el 1.08 % de las DEP
  (sin datos `_flt`); SCHED nunca falta.

## 4. Solapamiento entre fuentes (de las filas con BLOCK a ±30 s de la fuente de la fila, % que también está a ±30 s de la de la columna)

| si BLOCK ≈ | n filas | SCHED | IOBT | LOBT | EOBT_1 | AOBT_3 |
|---|---|---|---|---|---|---|
| SCHED_TIME | 113 447 | 100.0 | 92.4 | 92.0 | 76.3 | 9.8 |
| IOBT | 129 238 | 81.1 | 100.0 | 99.6 | 79.1 | 10.2 |
| LOBT | 128 789 | 81.1 | 99.9 | 100.0 | 79.5 | 10.2 |
| EOBT_1 | 139 702 | 61.9 | 73.2 | 73.2 | 100.0 | 11.0 |
| AOBT_3 | 217 734 | 5.1 | 6.0 | 6.0 | 7.0 | 100.0 |

Las copias de IOBT/LOBT/EOBT_1 son mayoritariamente la misma fila que la copia de SCHED (los planes coinciden con el horario).
AOBT_3 es la única fuente realmente distinta (solapa un 5–7 % con las de plan).

## 5. Por aeropuerto y por mes (atribución a τ = 30 s, % de DEP)

| aeropuerto | DEP | SCHED | IOBT | LOBT | EOBT_1 | AOBT_3 | ninguna |
|---|---|---|---|---|---|---|---|
| EDDF | 230 141 | 3.54 | 0.75 | 0.00 | 1.49 | 9.77 | 84.45 |
| EDDM | 167 334 | 4.99 | 0.89 | 0.00 | 1.97 | 11.58 | 80.58 |
| EGLL | 239 546 | 4.87 | 0.71 | 0.02 | 2.88 | 8.35 | 83.17 |
| EHAM | 247 951 | 3.47 | 0.96 | 0.00 | 1.23 | 11.14 | 83.19 |
| LEBL | 179 705 | 5.61 | 1.37 | 0.00 | 2.28 | 12.06 | 78.67 |
| LEMD | 212 242 | 4.46 | 1.20 | 0.00 | 2.04 | 12.03 | 80.28 |
| LFPG | 239 552 | 4.13 | 1.49 | 0.00 | 1.43 | 10.04 | 82.92 |
| LIRF | 160 704 | 18.10 | 1.64 | 0.00 | 4.39 | 8.10 | 67.76 |
| LSZH | 134 907 | 2.97 | 0.78 | 0.00 | 0.99 | 11.56 | 83.70 |
| LTFM | 272 965 | 5.15 | 1.78 | 0.01 | 0.00 | 3.79 | 89.28 |

Por mes la atribución es estable (rango ene–dic): SCHED 4.62–6.19 %, IOBT 1.04–1.28 %,
EOBT_1 1.28–2.28 %, AOBT_3 9.37–9.79 %, ninguna 80.87–83.46 %.

## 6. Gate A (prefijado en `protocol.md` §2)

| condición | valor | umbral | resultado |
|---|---|---|---|
| A1 · masa de copia no-SCHED (unión sin doble conteo; suma corregida) | 5.77 %; 16.81 % | ≥ 5 % | pasa |
| A2 · oráculo no-SCHED ±10 s sobre S4A_mix (V3), RMSE ponderado | 304.42 → 299.17 s (Δ -5.25 s) | Δ ≤ −15 s | **falla** |

A2 por mes: julio 350.2 → 344.7 s (18 295 filas, 9.59 %);
diciembre 234.4 → 229.4 s (16 628 filas, 10.04 %).
Bootstrap pareado (1 000 réplicas, bloques aeropuerto-día, estratificado por mes): Δ IC 95 % [-5.8, -4.7] s.
**Gate A: NO superado → la etapa termina aquí (protocolo §2).**

## 7. Por qué la hipótesis no es viable

1. **Las copias no-SCHED existen, pero caen donde V3 ya acierta.** Las filas que el oráculo corrige (9.59 % en julio,
   10.04 % en diciembre) tienen un RMSE de V3 de 199.4 / 153.2 s —por debajo de la media— y
   concentran solo el 3.11 % / 4.29 % del error cuadrático. Son mayoritariamente filas con BLOCK ≈ AOBT_3, y
   `MVT − AOBT_3` ya es una variable de V3. Aun sabiendo con certeza cuáles son, el RMSE baja ~5 s.
2. **El RMSE lo dominan filas que ninguna fuente visible explica.** En julio, 4 filas de fecha desplazada aportan el
   9.35 % del SSE de V3; la cola > 1 h que no es copia (358 filas, 0.19 %) aporta el
   12.05 %; las filas que copian alguna fuente (13.79 %) concentran el
   23.71 %, casi todo por las copias de SCHED (que V3 ya modela con p1), no por las no-SCHED.
   Ni siquiera una compuerta perfecta sobre todas las fuentes, SCHED incluida, baja de 260 s: 269.2 s.
3. **Robustez.** Con τ = 30 s el oráculo gana -6.0 s; con τ = 60 s, -12.0 s (y a ±60 s ya no
   son copias: el oráculo "acierta" el 25.7 % de las filas). Por aeropuerto solo 1 de 10 llega a −15 s
   (LEBL, -16.3 s). Ambas aproximaciones (masa sobre fondo local, oráculo a ±10 s) favorecen a la hipótesis; aun así no alcanza.

## 8. ¿Hay señal visible para identificar qué filas son copia? (AUC con tasa del grupo estimada fuera de mes)

| agrupación (visible en ranking) | grupos | copia SCHED | copia AOBT_3 | copia solo IOBT/LOBT/EOBT_1 | cualquier copia |
|---|---|---|---|---|---|
| airport | 10 | 0.693 | 0.636 | 0.659 | 0.663 |
| operator | 677 | 0.672 | 0.617 | 0.636 | 0.629 |
| flight_number | 45 718 | 0.717 | 0.647 | 0.670 | 0.669 |
| callsign | 44 825 | 0.711 | 0.648 | 0.670 | 0.667 |
| airport_stand | 2 725 | 0.712 | 0.657 | 0.677 | 0.675 |
| airport_runway | 70 | 0.697 | 0.643 | 0.663 | 0.667 |
| route | 6 077 | 0.714 | 0.652 | 0.675 | 0.672 |
| presence_pattern | 2 | 0.478 | 0.503 | 0.485 | 0.493 |
| aobt3_resolution | 30 | 0.695 | 0.641 | 0.663 | 0.666 |
| month | 12 | 0.500 | 0.500 | 0.500 | 0.500 |

La mejor agrupación (`flight_number` para `copy_SCHED`) llega a 0.717, solo
+0.024 sobre el aeropuerto por sí solo (0.693). Para las copias de AOBT_3 ninguna supera
0.657. El patrón de presencia no informa (≈ 0.5; solo hay 2 patrones: con o sin datos `_flt`). Nota metodológica: una primera versión usaba la tasa
leave-one-out por fila, que en agrupaciones de baja cardinalidad con una categoría dominante sesga el AUC muy por debajo de 0.5
(p. ej. 0.01 para el patrón de presencia); se sustituyó por la estimación fuera de mes antes de redactar este informe.

## 9. Resolución de los timestamps (% con segundos = 0, rango entre aeropuertos)

BLOCK 1.7–8.4 %, MVT 1.6–8.4 % (resolución de segundos);
SCHED 100.0–100.0 %, IOBT 100.0–100.0 %, LOBT 99.1–100.0 %,
EOBT_1 99.1–100.0 %, AOBT_3 83.8–100.0 % (minuto; EHAM 83.8 %);
ARVT_1 1.6–1.9 %, ARVT_3 22.6–41.9 %. La resolución de AOBT_3 como
agrupación no añade identificación sobre el aeropuerto (tabla §8, `aobt3_resolution`).

## 10. Anclas auxiliares (rotación del stand; no son copias)

El on-block de la llegada siguiente al mismo (aeropuerto, stand), visible en ranking, existe para el 79.6 %
de las DEP en 6 h, pero llega una mediana de 61 min después de BLOCK (p5 9 min) y solo en el
4.40 % de las DEP cae a 0–10 min: cota superior floja, no una fuente de BLOCK.

## 11. Conclusión

Las fuentes visibles explican una fracción real de BLOCK (≈ 15–18 % de las filas a ±10–30 s; ≈ 5.8 % de copias no-SCHED sin
doble conteo), pero esa fracción no es *material para el error*: incluso con una compuerta perfecta la mezcla de fuentes no-SCHED
mejora el RMSE ponderado en ~5 s. Conforme al protocolo, no se ejecutan las Etapas B–E ni se construye candidata.
