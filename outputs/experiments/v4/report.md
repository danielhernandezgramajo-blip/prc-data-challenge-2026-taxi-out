# V4 — Informe final: parada en la Etapa A

Generado por `src/v4_report.py`. Protocolo: `protocol.md` (guardado antes de auditar y de entrenar). Auditoría: `source_audit.md`.

## Veredicto

**La hipótesis de selección de fuentes no es viable.** BLOCK sí coincide con fuentes visibles en una fracción real de filas
(≈ 15–18 %; ≈ 5.8 % son copias no-SCHED sin doble conteo), pero esas filas son las que V3 ya predice bien. Un oráculo que
conoce exactamente qué filas copian una fuente no-SCHED mejora el RMSE ponderado jul/ene en **-5.2 s**
(IC 95 % [-5.8, -4.7]), frente a los −15 s exigidos por el Gate A. Ni siquiera una compuerta perfecta
sobre todas las fuentes, SCHED incluida, baja del umbral de parada de 260 s (269.2 s). Conforme al protocolo: no se
ejecutaron las Etapas B–E, no se entrenó ningún modelo V4, no existe `candidate_v4`, no se subió nada, no se hizo push y el
leaderboard no se usó para ninguna decisión.

## 1. Qué se ejecutó y qué no

| etapa | estado |
|---|---|
| Protocolo prerregistrado (`protocol.md`) | hecho, antes de cualquier cálculo |
| A · auditoría de fuentes (cobertura, atribución, confusión, aeropuerto, mes, operador, vuelo, callsign, stand, pista, ruta, presencia, resolución) | hecho → **Gate A no superado** |
| Oráculos suplementarios (no deciden el gate; etiquetados como cota optimista) | hecho, para cuantificar dónde está el margen |
| B · features V4 · C · C0–C5 · D · LOJO/LOJuO, 3 semillas · E · candidata | **no ejecutado** (parada prefijada) |

## 2. RMSE ponderado (julio 2025 y diciembre 2025 como proxy de enero; pesos del ranking 0.557 / 0.443)

C1–C5 no se ejecutaron. Como C0 se usa la predicción OOF ya existente de V3 (S4A_mix, folds de origen creciente), no un
reentrenamiento LOJO/LOJuO. Las filas siguientes son **oráculos**: conocen el target para decidir la compuerta; son cotas de lo
alcanzable con una compuerta perfecta, no resultados realizables. B→E son acumulativas.

| configuración | julio | diciembre | RMSE_w | IC 95 % RMSE_w | Δ vs C0 [IC 95 %] |
|---|---|---|---|---|---|
| C0 de referencia: S4A_mix (V3), OOF origen creciente | 350.2 | 234.4 | **304.4** | [279.3, 332.2] | — |
| Oráculo Gate A: + fuentes no-SCHED perfectas (±10 s) | 344.7 | 229.4 | **299.2** | [273.5, 327.1] | -5.2 [-5.8, -4.7] |
| ↳ sensibilidad ±30 s | 344.1 | 228.3 | **298.4** | [272.7, 326.4] | -6.0 [-6.5, -5.5] |
| ↳ sensibilidad ±60 s (ya no son copias) | 338.3 | 221.8 | **292.5** | [266.5, 320.8] | -12.0 [-12.9, -11.1] |
| Compuerta perfecta sobre TODAS las fuentes (incl. SCHED) | 305.9 | 214.4 | **269.2** | [247.9, 293.3] | -35.2 [-44.9, -26.6] |
| + fecha desplazada perfecta | 286.6 | 214.2 | **257.0** | [241.5, 273.4] | -47.4 [-66.4, -31.8] |
| + resto con la rama normal pura (sin masa p1) | 258.6 | 208.1 | **237.6** | [227.1, 249.8] | -66.9 [-89.2, -46.0] |
| + capacidad/escala Etapa 3 en el grueso (aprox.) | 250.9 | 201.0 | **230.2** | [219.6, 242.5] | -74.3 [-96.6, -53.4] |
| + cola > 1 h perfecta (suelo) | 218.0 | 189.2 | **205.7** | [200.4, 211.9] | -98.7 [-123.7, -75.9] |

Bootstrap: 1000 réplicas pareadas, bloques aeropuerto-día, estratificado por mes, recombinado con los pesos; semilla 42.
La razón de MSE de la Etapa 3 (0.924) viene del diagnóstico V3 (`exp_capacity.json`).

## 3. Estabilidad por aeropuerto (RMSE ponderado jul/dic de cada aeropuerto; Δ de los oráculos frente a C0)

| aeropuerto | C0 (V3) | Δ oráculo Gate A | Δ todas las fuentes | Δ hasta C2 |
|---|---|---|---|---|
| EDDF | 196.4 | -1.3 | -2.5 | -2.6 |
| EDDM | 218.4 | -4.5 | -7.0 | -6.8 |
| EGLL | 298.2 | -4.9 | -8.7 | -9.0 |
| EHAM | 179.1 | -1.7 | -2.3 | -2.6 |
| LEBL | 244.9 | -16.3 | -43.9 | -44.7 |
| LEMD | 193.7 | -6.9 | -10.4 | -10.8 |
| LFPG | 280.0 | -6.0 | -8.4 | -8.5 |
| LIRF | 710.6 | -7.8 | -159.6 | -391.1 |
| LSZH | 198.2 | -1.7 | -2.5 | -3.0 |
| LTFM | 280.4 | -4.3 | -8.9 | -8.9 |

El oráculo del Gate A no llega a −15 s en 9 de 10 aeropuertos. El que más ganaría con una compuerta perfecta es
LIRF (-159.6 s con todas las fuentes; -391.1 s hasta C2): es un problema de identificar la copia de
SCHED y la fecha desplazada, que V3 ya modela, no de añadir fuentes.

## 4. Columnas nuevas con señal real

Ninguna con señal material para separar fuentes. AUC fuera de mes (`source_audit.md` §8): número de vuelo, callsign, ruta y
stand ≈ 0.71 para copias de SCHED (≈ +0.02 sobre el aeropuerto solo), ≤ 0.66 para copias de AOBT_3; patrón de presencia ≈ 0.5;
resolución de AOBT_3 no añade nada. Sin Etapas B/C no se midió la importancia de las features V4 en un modelo.

## 5. ARVT_1 / ARVT_3

No evaluadas: solo entraban como features de la variante withARVT en las Etapas B/C, que no se ejecutaron. No son fuentes de
BLOCK (son tiempos de llegada en destino, posteriores al despegue) y no intervienen en la auditoría ni en los oráculos.
No hay resultados "con y sin ARVT" que reportar.

## 6. Desviaciones y correcciones respecto al protocolo

- AUC de agrupaciones: el leave-one-out por fila estaba sesgado (§8 de la auditoría); se sustituyó por la tasa estimada fuera de
  mes antes de redactar. No afecta al gate.
- Oráculos suplementarios (umbrales ±30/60 s y escalera B→E) añadidos tras conocer el gate, solo para diagnóstico; el gate se
  decidió con la definición prerregistrada (±10 s), que no cambió.
- El oráculo del gate pasó a alinear filas con `align_by_id` (falla si algún MVT_ID falta o se duplica); el resultado es
  idéntico al de la primera ejecución.
- `models/experiments/v4/` quedó vacío: no se entrenó nada.

## 7. Qué implica (dónde está el margen, según los oráculos)

Con compuerta perfecta: saber qué filas copian una fuente (casi todo SCHED) vale -35.2 s; la fecha desplazada,
-12.2 s más; quitar la masa p1 que la mezcla reparte por prudencia en las filas normales, -19.5 s más; la
capacidad/escala de la Etapa 3, ≈ -7.4 s; la cola > 1 h, -24.4 s. Todo ese margen depende de **identificar**
qué filas son artefacto, no de qué fuente se copió, y las señales visibles para identificarlo son débiles (AUC ≤ 0.72 por
agrupación; el clasificador de V3 ya combina las útiles). Con compuerta perfecta sobre las fuentes el techo es 269.2 s;
una compuerta real queda por encima. La banda 220–240 s solo se alcanza con identificación perfecta de todos los artefactos.

## 8. Integridad y cumplimiento

Verificación de solo lectura. Entorno: copia de trabajo git (autoritativa).

| archivo / conjunto | resultado |
|---|---|
| submissions/candidate_v1.parquet | ok |
| submissions/candidate_v2.parquet | ok |
| submissions/candidate_v3.parquet | ok |
| submissions/candidate_v1.json | ok |
| submissions/candidate_v2.json | ok |
| submissions/candidate_v3.json | ok |
| outputs/_mac_tracked_md5.txt | ok |
| 14 Parquet del challenge (sha256 conjunto) | ok |
| 43 archivos de bundles finales (md5 conjunto) | ok |

Sin subidas al bucket, sin push, sin datos externos, sin uso del leaderboard. MVT_ID solo para alinear y FLIGHT_ID solo para
enlazar (no se usó). Los 14 Parquet, candidate_v1–v3 y sus JSON, los bundles finales y `_mac_tracked_md5.txt` no se tocaron.

## 9. Archivos

`src/v4_sources.py`, `src/v4_audit.py`, `src/v4_oracles.py`, `src/v4_report.py`, `tests/test_v4.py` y en esta carpeta:
`protocol.md`, `source_audit.md`, `report.md`, `decision.json`, `gate_a.json`, `oracles.json`, `oracle_by_airport.csv`,
`source_coverage.csv`, `source_attribution_by_airport.csv`, `source_attribution_by_month.csv`, `source_confusion.csv`,
`source_attribution_by_group.csv`, `source_group_informativeness.csv`, `timestamp_resolution.csv`, `aux_stand_anchors.csv`.
