# V6 — Protocolo (2026-10-05)

Base: V5a (V4b C2 con rama normal 0.5·CatBoost + 0.5·LightGBM; validada en `outputs/experiments/v5/protocol.md` §2).
Reglas de siempre: los 14 Parquet no se tocan; sin subidas ni push; MVT_ID solo para alinear; el leaderboard no se usa para
elegir nada. Este proyecto ya tiene 2 submissions (`optimistic-dragon_v1` = V3, `optimistic-dragon_v2` = V4b); la política
a priori de `outputs/experiments/v3/protocol_amendments.md` decía "máximo dos". Una tercera submission es decisión de
Daniel y queda registrada aquí como enmienda, no como algo decidido por el pipeline.

## 1. Regla de hueco LOBT (exploratoria → validada fuera de mes)

Origen (honesto): se buscó por qué V4b no mejoró a V3 en el leaderboard comparando las predicciones de V3 y V4b fila a fila
y buscando análogos en 2025. La forma de la regla (umbral MVT − LOBT > 2 h, recta en MVT − LOBT) se eligió mirando los 12
meses de 2025, incluidos enero y julio; en la validación solo los coeficientes se ajustan fuera de mes. Es decir, la
validación es fuera de muestra para el ajuste, no para la elección de la forma.

- Definición: si MVT − LOBT > 7200 s → ŷ = max(0, a + c·(MVT − LOBT)); (a, c) por mínimos cuadrados en las filas de
  entrenamiento con MVT − LOBT > 7200 s e y > 0. Final: 143 filas de 2025, a ≈ −1228, c ≈ 1.015.
- Por qué existe: cuando el avión sale del calzo cerca de su LOBT y despega horas después, el plan NM ya no se actualiza;
  en 2025 las 143/143 filas con MVT − LOBT > 2 h tienen taxi > 1 h, con y ≈ MVT − LOBT − ~1000 s. Los árboles no extrapolan.
- Validación (rama V5a; regla ajustada solo con los 9 meses de entrenamiento de cada fold):
  ene/jul RMSE_w 333.93 → 322.43 (Δ −11.49 s, IC 95 % bootstrap aeropuerto-día [−34.09, −0.28]);
  enero −0.69, julio −21.27; febrero −0.43, agosto −0.40. Sin las 3 filas de mayor influencia: −0.08 / −0.21 / −0.12 / −0.07
  (mejora pequeña pero en los 4 meses). Umbral 1.5 h: −10.16; 3 h: −10.76 (robusto al umbral).
  Variante con MVT − AOBT_3 > 1 h: +14.99 s (empeora) → descartada. CV por mes en las 143 filas: solo L (RMSE 1320) mejor que
  A (6323), L+A (1336) o por tramos (1311, no significativo) → se queda la recta en L.
- Ranking: 93 filas afectadas (48 de EHAM 3–7 ene 2026, AOBT_3 a 1.5 h = tope del NM; 1 de 24 h en LIRF 17‑jul‑2026).
  Riesgo conocido: en EHAM 5‑ene‑2025 (nieve) y ≈ 1.25 h con L ≈ 2.1–2.2 h → la recta puede pasarse ~0.5 h en las filas de
  EHAM 2026 (≈ +1 s en el peor caso razonable). La fila de 24 h domina: si su taxi fuera normal (~20 min) la regla costaría
  ≈ +30 s; si se parece al único análogo de 2025 (y = 24.2 h) ahorra ≈ 35–45 s.

## 2. Compuerta por prefijo de aerolínea en LIRF sin datos NM (configuración fijada antes de evaluarla fuera de mes)

Hallazgo mirando la tabla de todo 2025 (incluye ene/feb/jul/ago, así que la idea no es ciega; sí lo es el ajuste fuera de
mes y la configuración, que se fija aquí antes de calcular ningún resultado de validación): en LIRF, las salidas sin plan NM (EOBT y LOBT nulos; operador = UNK, así que
`te_apt_operator` no aporta) tienen una tasa de BLOCK = SCHED muy distinta según el prefijo OACI del vuelo (`FLIGHT_mvt`):
ISR/LAV/ETH/TKJ/TWB 100 %, AEZ 84 %, WZZ 74 % frente a KLM/EWG/ELY 0 %, EZY 8 %, ITY 12 %, RYR 33 % (s > 1 h). Probable
proxy del agente de handling. La compuerta V4b ve el número de vuelo completo (TE) pero no el prefijo.

Prueba fijada ahora (una sola configuración):
- Filas: LIRF, EOBT y LOBT nulos, fuera de la celda de fecha desplazada, s = MVT − SCHED > 1800 s.
- Modelo: regresión logística L2 (C = 1.0) con dummies de banda de s (<1 h, 1–2, 2–4, 4–8, 8–16, >16 h) y dummies de
  prefijo (prefijos con < 5 filas → "OTRO"); etiqueta clase 1 (|y − s| ≤ 10 s). Ajuste con los mismos 9 meses de
  entrenamiento de cada fold.
- Predicción: p1 = 0.5·p1_V4b + 0.5·p1_prefijo en esas filas; el resto no cambia.
- Base de comparación: V5a + regla LOBT (§1). Regla de adopción: ΔRMSE_w ≤ −1 s en ene/jul y mejora en enero y en julio;
  febrero/agosto se informan. Si no se cumple, se descarta y V6 = V5a + regla LOBT.

### Resultado §2 (2026-10-05) → DESCARTADA

ΔRMSE_w = +0.42 s (IC 95 % [−2.12, +2.94]); enero −1.13, julio +1.86 (febrero +0.61, agosto −2.24). La log-loss de la
compuerta en esas filas empeora en los 4 meses (p. ej. julio 0.379 → 0.431): la compuerta V4b ya recoge el efecto de la
aerolínea (TE del número de vuelo), y el prefijo solo añade ruido. No cumple la regla → V6 = V5a + regla LOBT.
(`outputs/experiments/v6/prefix_gate_check.json`)

## 3. Candidata V6 (2026-10-05)

`submissions/candidate_v6.parquet` — sha256 `3fe341e44a41d1021b43809b604ec77824dfd9215801f1bf4bc694b0f714b84a`,
344 841 filas, MVT_ID_mvt Float64 / TAXITIME_SEC_mvt Int32, comprobaciones OK al reabrir. V6 = V5a + regla LOBT
(a = −1228.23, c = 1.01532, n = 143); compuerta por prefijo desactivada (§2 no pasó).
- Frente a V5a cambian exactamente las 93 filas de la regla (88 suben, 5 bajan; mediana +2715 s): EHAM 48, LFPG 16,
  LIRF 10, EGLL 9, EDDF 5, LTFM 2, LSZH 2, EDDM 1. La mayor: LIRF 17‑jul‑2026 (MVT − LOBT = 24.1 h, MVT − AOBT_3 = 24.0 h)
  3663 → 86 922 s.
- Frente a V4b: cambia la rama normal en casi todas las filas (media |Δ| 21.8 s, p99 134 s) + las 93 de la regla.
- Construcción: en la nube (features idénticas al Mac por hash; predicciones por semilla de V4b del Mac, que reproducen
  candidate_v4b exactamente; LightGBM 12 meses entrenado en la nube con `src/v5_lgb.py`, mismo código y semilla).
  Reproducible en el Mac con `bash src/v6_run_candidate.sh` (usa la caché LightGBM, tarda segundos).

Lectura honesta del riesgo:
- Fila de 24 h: en 2025, cuando MVT − AOBT_3 > 2 h y MVT − LOBT > 2 h a la vez (sin LOBT nuevo tras el off-block), 6/6
  filas tienen y > 1 h, y la única con hueco > 6 h tuvo y = 24.2 h ≈ MVT − AOBT_3. Cuando el AOBT_3 es viejo pero hay LOBT
  reciente (L ≤ 2 h), y es corto (11/13 < 1 h): por eso la regla usa L y no A.
- Si esa fila se comporta como 2025: ≈ −35 a −45 s frente a V4b; si su taxi real fuera normal: ≈ +30 s.
- Resto de filas de la regla: ≈ −2 s esperado (−0.1/−0.2 s por mes en 2025 sin las filas extremas); EHAM 2026 puede
  pasarse si se parece al día de nieve de 2025 (≈ +1 s en ese escenario).
- Mezcla LightGBM: −1.41 s (validada en §2 de V5).
