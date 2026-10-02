# V4 — Protocolo (escrito y guardado ANTES de la auditoría de fuentes y de cualquier entrenamiento V4)

Fecha: 2026-10-01. Estado de partida: HEAD `056bbc4`, árbol limpio salvo `outputs/_mac_tracked_md5.txt` (sin rastrear,
se conserva). Hashes protegidos: candidate_v1 `d482a42b…`, candidate_v2 `e1db051a…`, candidate_v3 `165c6e29…`,
candidate_v{1,2,3}.json `8b6a4aec…`/`6cb17632…`/`cde6ea19…`, `_mac_tracked_md5.txt` `301e9aea…`; 14 Parquet (sha256
conjunto en la nube `1df017af…`); 43 archivos de bundles finales (md5 conjunto `28fd7b74…`). Todo se re-verifica al final.

## 0. Reglas

1. Solo los 14 Parquet del challenge. Sin datos externos. Sin leaderboard: el score público 317.9925 es conocido y NO se
   usa para elegir clases, tolerancias, pesos, hiperparámetros ni features. Ninguna subida al bucket. Ningún push.
2. `MVT_ID_mvt` solo para alinear filas; `FLIGHT_ID_mvt` solo para enlazar filas; nunca como valores predictivos.
3. Variables posteriores al despegue (`ARVT_1_flt`, `ARVT_3_flt`) solo en la variante "withARVT"; la variante primaria es
   "noARVT". Se reportan ambas para la finalista.
4. Columnas en ranking: se usan todas las columnas presentes en las filas DEP del ranking salvo `BLOCK_TIME_UTC_mvt` y
   `TAXITIME_SEC_mvt` (ocultas). Filas ARR del ranking: se usan sus columnas visibles (incluido su BLOCK, que en ARR es
   visible) para features de contexto, calculadas igual en training y ranking.
5. Todo lo nuevo en `src/v4_*`, `outputs/experiments/v4/`, `models/experiments/v4/`, `tests/test_v4.py`. No se
   sobrescribe nada de V1–V3.

## 1. Información previa ya conocida (declaración de no-ceguera)

Antes de este protocolo ya se había observado sobre training 2025 (V2/V3 y diagnóstico post-leaderboard):
BLOCK = SCHED (±10 s) en 4.8 % de las DEP; en julio 2025, |y − (MVT − X)| ≤ 10 s / ≤ 65 s para X = SCHED 4.19/9.83 %,
EOBT_1 4.79/12.74 %, LOBT 4.80/11.94 %, IOBT 4.83/12.00 %, AOBT_3 7.94/25.86 %; en clase 0 de julio, BLOCK − AOBT_3 cae
en ±60 s el 20.6 % y en ±300 s el 73.5 %, con un pico estrecho en ±10 s sobre un fondo ancho. Estas cifras motivan la
hipótesis; las decisiones de diseño de abajo se fijan sin mirar resultados de modelos V4.

## 2. Etapa A — Auditoría de fuentes y Gate A

Anclas: S = {AOBT_3, LOBT, EOBT_1, IOBT, SCHED} × desplazamiento {0, −86 400, +86 400} (15 anclas). Para cada DEP de
training: `d_a = BLOCK − (ts_a + shift_a)` en segundos. Cobertura en ±10/30/60/120/300 s por ancla y unión; fuente más
cercana, distancia, segunda más cercana, ambigüedad (≥ 2 anclas distintas dentro de la tolerancia), fuente ausente,
desplazamiento de fecha. Desgloses por aeropuerto, mes, operador, número de vuelo, callsign, stand, pista, ruta y
presencia/resolución de cada timestamp (segundos == 0).

"Masa de copia" de un ancla: exceso de filas con |d| ≤ 10 s sobre el fondo local, estimado con la densidad por segundo en
los hombros 30 < |d| ≤ 120 s: `copy = (N(|d|≤10) − 20·dens_hombro) / N_filas`. Distingue copias reales de cercanías
casuales de dos medidas del mismo evento. Se calcula también tras centrar d con la moda por (aeropuerto, ancla)
("corrección residual").

**Gate A (prefijado).** La hipótesis se considera viable solo si se cumplen las dos condiciones:
- A1 (proporción): la masa de copia de las anclas distintas de SCHED-sin-desplazar (ya modelada por V3) suma ≥ 5 % de
  las DEP de 2025, con o sin corrección residual.
- A2 (relevancia para el error): un ORÁCULO que sustituye la predicción OOF de S4A_mix (V3) por `MVT − ancla` en las filas
  con alguna ancla no-SCHED dentro de ±10 s reduce el RMSE ponderado (julio 2025 y diciembre 2025 como proxys de julio y
  enero, pesos del ranking) en ≥ 15 s. Es una cota optimista (el oráculo conoce qué filas son copia).
Si falla A1 o A2: la etapa termina, se documenta por qué y NO se construye candidata.

## 3. Etapa B — Features V4 (calculadas igual en training y ranking, sin target)

Base V1 (29) + congestión V2 (14) + `days_after_sched` (V3) + nuevas: diferencias entre todos los pares de timestamps
visibles; disponibilidad y resolución de cada timestamp; mismatches mvt/flt (ADEP, ADES/ADES_FILED, tipo, regla,
FLIGHT_mvt vs CALLSIGN); FLIGHT_mvt y CALLSIGN_flt categóricas (nativas de CatBoost) y su frecuencia por mes (sin
target; comparadas); minuto del día, bins 5/10/15/30 min, día del año, semana, cíclicas; interacciones categóricas
(aeropuerto×stand×pista, aeropuerto×operador×vuelo, ruta×aeronave, aeropuerto×fuente disponible, pista×bin 15 min,
operador×disponibilidad/resolución de AOBT_3); secuenciales por pista (tiempo al despegue anterior/siguiente), salidas con
AOBT_3 anterior y MVT posterior (cola estimada), llegadas intercaladas, rotación del stand (llegada previa: minutos desde
su on-block/aterrizaje, su retraso y taxi-in; llegada siguiente al mismo stand: minutos hasta su on-block visible),
features de escala de la Etapa 3. ARVT_1/3 (solo withARVT): duración real/planificada, ARVT_3 − ARVT_1, consistencia
con AOBT/EOBT.

## 4. Etapa C — Mezcla de expertos

Clases del gating: cada ancla con ≥ 500 filas atribuidas en 2025 (tolerancia τ = 30 s; empate → ancla más cercana;
coincidencias exactas entre anclas → precedencia SCHED, IOBT, LOBT, EOBT_1, AOBT_3, que no altera la predicción porque las
anclas coinciden) + "none". Anclas con < 500 filas se funden en "none". La clase de fecha desplazada de V3 (13 filas en
2025) no es aprendible por un GBM: su probabilidad p_ds la da el RegimeClassifier2 de V3 (modo anidado) entrenado en el
mismo outer-train, y su experto es `none − 86 400`.

`E[BLOCK|x] = (1 − p_ds)·Σ_k p_k·(ts_k + r̂_k) + p_ds·(BLOCK_none − 86 400)`; `TAXI = MVT − E[BLOCK|x]`; recorte a ≥ 0 en
la submission. Fuentes ausentes en una fila: p_k = 0 y renormalización.

- C0: S4A_mix (V3) reentrenado en cada outer-train, mismas filas.
- C1: gating + expertos de fuente sin corrección (r̂_k = 0) + experto none (config V3: 0.5·directo + 0.5·residual con
  features V1+V2 de superficie, solo filas none ≤ 3 h).
- C2: C1 con r̂_k = CatBoost de residuo por fuente (filas atribuidas).
- C3: C2 con features V4 completas (noARVT) en gating y expertos, y experto none de capacidad Etapa 3
  (3000 it, depth 8, lr 0.05).
- C4: C3 con expertos none por aeropuerto promediados 50/50 con el global.
- C5: C3 con el experto none promediado 50/50 con LightGBM (si se instala sin romper el entorno).
Gating: CatBoost/LightGBM multiclase; la clase none se submuestrea al 25 % con corrección exacta del prior; calibración por
temperatura en un mes interno (ver D). Orden de ejecución: C0–C2 → C3 → C4/C5; si tras C3 el RMSE ponderado está a más de
25 s por encima de 260 en ambos meses, C4/C5 no pueden cerrar la brecha (sus ganancias esperadas son de pocos segundos) y
se documenta la parada.

## 5. Etapa D — Validación

Outer: LOJO (entrena feb–dic 2025, evalúa enero 2025) y LOJuO (entrena ene–jun + ago–dic, evalúa julio 2025). Control para
finalistas: folds de origen creciente julio y diciembre (V2). Mes interno de calibración del gating (excluido del ajuste del
gating, no de los expertos): febrero para LOJO, agosto para LOJuO. Nada calculado sobre el mes evaluado ve su target.
Evaluación sobre TODAS las DEP del mes (target original, extremos incluidos).

RMSE ponderado: `RMSE_w = sqrt(0.4429·MSE_ene + 0.5571·MSE_jul)` con MSE = SSE/n por mes y pesos = filas del ranking
(152 719 / 192 122). Métricas: RMSE total, por mes, por aeropuerto, por régimen/fuente, grueso (≤ 1 h), 1–3 h, > 3 h; MAE,
MedAE; log-loss y calibración del gating; concentración del SSE; bootstrap pareado por bloques aeropuerto-día (1 000,
SEED 42, estratificado por mes y recombinado con los pesos); 3 semillas (42, 7, 2025) para finalistas.

Gates de resultado: RMSE_w estable = estimación puntual y extremo superior del IC 95 % bootstrap por debajo del umbral, y
el máximo de las 3 semillas también. ≥ 260 → parar y documentar. 240–260 → conservar como mejora, sin candidata. 220–240
con mejora frente a C0 en ambos meses, en ≥ 8/10 aeropuertos y P(mejora) ≥ 0.95 → entrenar el modelo final (Etapa E).
La mejora debe mantenerse en noARVT; withARVT solo se conserva si es legítima y claramente mejor.

## 6. Etapa E — Candidata (solo si pasa los gates)

Entrenar con todo 2025, generar `submissions/candidate_v4.parquet` (344 841 filas, esquema y orden del template, sin
nulos/infinitos/negativos) y `candidate_v4.json` con `uploaded_to_bucket: false`. Sin renombrar ni subir.
