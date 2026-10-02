# V4b — Protocolo (escrito y guardado ANTES de construir features nuevas y de entrenar)

Fecha: 2026-10-01. Base: commit `c68179e` (V4, resultado negativo válido, se conserva) + corrección del prior fuera de mes en
`src/v4_audit.py` (commit aparte). Objetivo: comprobar honestamente si el problema principal de V3 es **una compuerta mal
calibrada** y/o **un modelo normal insuficiente**. No es una búsqueda de top 3.

## 0. Reglas

1. Solo los 14 Parquet. Sin datos externos. Sin leaderboard (el 317.99 público no se usa para nada). Sin subidas, sin push,
   sin candidata hasta presentar resultados. MVT_ID solo para alinear; FLIGHT_ID no se usa.
2. Todo lo nuevo en `src/v4b_*`, `outputs/experiments/v4b/`, `models/experiments/v4b/`, `tests/test_v4b.py`.
3. ARVT_1/ARVT_3 (posteriores al despegue) quedan fuera de la variante primaria; solo entran en la ablación de la etapa costosa.
4. Las cotas de oráculo de V4 (p. ej. 230.2 s, 205.7 s) son **no alcanzables** y no se usan como objetivo ni como referencia de
   éxito.

## 1. Información previa (declaración de no-ceguera)

- Julio 2025 se ha usado muchas veces (fold de desarrollo V2/V3, oráculos V4): no es un holdout limpio. Enero 2025 nunca fue
  mes de evaluación (siempre estuvo en entrenamiento): es la comprobación más limpia. Se reportan ambos por separado.
- Ya conocido de V3 (OOF de origen creciente, julio/diciembre): AUC de clase 1 ≈ 0.86–0.88; p1 medio ≈ 0.18–0.20 en filas de
  clase 1 y ≈ 0.03–0.04 en filas de clase 0; la masa p1 en filas normales sube el RMSE de clase 0 (julio 299.5 frente a 271.2
  con la rama normal pura); la clase 2 vive casi entera en la celda (LIRF, sin AOBT_3, fecha MVT > fecha SCHED: 68 filas, 54 de
  clase 1, 11 de clase 2), su p2 fuera de la celda es despreciable (media ~1e-5) y su coste está dentro de la celda.
- Etapa 3 (diagnóstico V3, julio clase 0 ≤ 1 h): 241.9 → 232.6 con capacidad + escala + FLIGHT. Expectativa honesta antes de
  correr: rama normal −5/−8 s; compuerta, de 0 a −10 s. Llegar a −15 s exige que funcionen las dos.

## 2. Validación

- Outer: **LOJO** (evalúa enero 2025) y **LOJuO** (evalúa julio 2025). Se evalúa sobre TODAS las DEP del mes (target original).
- Meses internos de calibración: **febrero y agosto** (los mismos para ambos outer). Componentes entrenados con los 9 meses
  restantes (LOJO: mar–jul + sep–dic; LOJuO: ene + mar–jun + sep–dic), con `drop_nonpositive` como en V1–V3.
- `RMSE_w = sqrt(0.4429·MSE_ene + 0.5571·MSE_jul)` (pesos = filas del ranking 152 719 / 192 122).
- Semillas 42, 7, 2025 para todos los componentes de C0, C1 y C2. Se ejecuta primero la semilla 42 completa.
- Métrica primaria: RMSE_w del **ensamble de 3 semillas** (media de predicciones) de cada configuración. Se reportan además las
  3 semillas por separado (mínimo/máximo de Δ).

## 3. Features V4b (training y ranking con el mismo código; sin target ni BLOCK de las DEP)

Base = V1 + congestión V2 (+ relativas, ajustadas solo con los meses de entrenamiento) + `days_after_sched`. Nuevas:
- plan: `eobt1_minus_sched`, `lobt_minus_sched`, `iobt_minus_sched`, `aobt3_minus_eobt1`, `eobt1_minus_lobt`, `lobt_minus_iobt`;
- resolución: `aobt3_sec0`, `mvt_sec0`;
- escala (llegada previa al mismo aeropuerto+stand antes del despegue, ≤ 24 h; columnas ARR visibles en ranking):
  `min_since_prev_arr`, `prev_arr_delay_min`, `prev_arr_taxi_in`, `prev_arr_copy` (BLOCK = SCHED en esa llegada);
- contexto de artefacto en llegadas del aeropuerto ±60/±180 min: `arr_copy_rate_pm60`, `arr_copy_rate_pm180`, `arr_n_pm60`;
- frecuencia sin target: `flight_n_month`;
- categóricas: `FLIGHT_mvt`, `CALLSIGN_flt` e interacciones `apt_stand`, `apt_runway`, `apt_operator`, `route`.

## 4. Configuraciones mínimas

- **C0 = V3 (S4A_mix) reentrenado** con los mismos 9 meses: `normal_C0 = 0.5·direct_normal_s + 0.5·residual_normal_s` (A_surface,
  1000 it, configuración V3) y RegimeClassifier2 (modo anidado, α = 20). `ŷ = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal)`.
- **C1 = compuerta nueva, rama normal de C0.** Clasificador multivariable de clase 1 (CatBoost Logloss, features V4b, 1000 it,
  depth 6, lr 0.08, max_ctr_complexity 2; negativos submuestreados al 30 % con peso 1/0.3, semilla fija) + etapa 2 orientada al
  RMSE (§5) + fecha desplazada aparte (§6).
- **C2 = C1 + rama normal nueva**: CatBoost RMSE, features V4b, 3000 it, depth 8, lr 0.05, max_ctr_complexity 2, border 128,
  clase 0 y 0 < y ≤ 3 h. Regla de cómputo prefijada: si el primer ajuste tarda > 60 min, todas las unidades pasan a 2000 it con
  lr 0.075 (decisión tomada solo con el tiempo, antes de ver métricas). La rama normal de C2 se elige en los meses internos entre
  {nueva sola, 50/50 con normal_C0}.

## 5. Etapa 2: combinación orientada al RMSE, validada de forma anidada

En las filas fuera de la celda de fecha desplazada, con `b` = rama normal, `p` = probabilidad de clase 1 y `Δ = (MVT − SCHED) − b`:
- `S` (mezcla suave con p tal cual): `ŷ = b + p·Δ`;
- `S_T` (Platt por log-loss): `p' = σ(a·logit p + c)`;
- `S_R2` (calibración por RMSE): `(a, c)` que minimizan `Σ (y − b − σ(a·logit p + c)·Δ)²`;
- `S_R3`: como S_R2 con un término `d·log(1 + |Δ|/600)`;
- `H` (compuerta dura): `ŷ = MVT − SCHED` si `p > t`, si no `b`; `t` minimiza el SSE;
- `D` (regresión directa de la corrección): CatBoost RMSE (300 it, depth 6) de `r = y − b` con features
  `[logit p, MVT − SCHED, b, Δ, |Δ|, has_aobt3, days_after_sched, mvt_minus_aobt3, hour_utc, ADEP]`; `ŷ = b + r̂`;
- `Q` (corrección estructurada): CatBoost RMSE de `t = clip((y − b)/Δ, −1, 2)` con peso `Δ²` en filas con `|Δ| ≥ 60 s`;
  `ŷ = b + clip(q̂, 0, 1)·Δ` (minimiza el SSE final de la forma `b + qΔ`).
Fuentes de p: `p1_C0` (clasificador de V3; aísla "solo recalibrar") con S, S_T, S_R2, S_R3, H; `p1_new` con las 7 variantes.
**Anidado**: cada variante se ajusta en febrero y se evalúa en agosto y viceversa (SSE interno cruzado, todas las filas internas).
La variante primaria de C1 y de C2 (y la rama normal de C2) es la de menor SSE interno sumado sobre outer × semillas; después se
reajusta con febrero + agosto y se aplica al mes outer. Nunca se mira el mes outer para elegir. Las demás variantes se reportan
como diagnóstico. Filas de clase 2 excluidas del ajuste de D y Q.

## 6. Fecha desplazada (clase 2) aparte

Dentro de la celda "desplazada ≥ 1 día y sin AOBT_3" se usan las probabilidades bayesianas por celda/banda de RegimeClassifier2
(modo anidado) con la rama normal de cada configuración. Fuera de la celda, **p2 = 0** (no se reparte masa sobre la población).
Su contribución se reporta aparte (enero 2025 tiene 1 fila de clase 2; julio 2025, 4: resultado muy ruidoso).

## 7. Métricas y comparación

Por configuración, mes y semilla: RMSE total; clase 0 (normal), clase 0 ≤ 1 h, 1–3 h, > 3 h (cola); clase 1; clase 2; MAE;
reparto del SSE; RMSE por aeropuerto; para los clasificadores AUC, log-loss y Brier ponderado por Δ² (lo que importa al RMSE).
Bootstrap pareado (1 000 réplicas, bloques aeropuerto-día, estratificado por mes, recombinado con los pesos, semilla 42) para
C1 − C0, C2 − C0 y C2 − C1 sobre los ensambles.

## 8. Regla de decisión

- Se continúa a la etapa costosa **solo si** el ensamble C2 mejora a C0 en **≥ 15 s** de RMSE_w (estimación puntual). Si pasa
  pero el IC 95 % cruza 0, o una semilla no mejora, se marca como frágil.
- C1 − C0 responde "¿compuerta mal calibrada?"; C2 − C1 responde "¿rama normal insuficiente?".
- Etapa costosa (solo si pasa): ablación con/sin ARVT; adaptación no supervisada a la distribución del ranking, simulada en la
  validación adaptando a las features (sin target) del mes outer; más capacidad (max_ctr_complexity 3, ensamble con LightGBM).
  Tampoco genera candidata sin presentar resultados.

## 9. Enmiendas (decididas antes de ver ninguna métrica)

- **A1 (cómputo, solo con tiempos)**. Primer ajuste de la rama normal nueva (LOJuO, s42): con max_ctr_complexity 2 y las 17
  categóricas, 2.3 s/iteración (≈ 1.9 h por ajuste de 3000 it; ≈ 1.3 h incluso con la regla 2000 it/lr 0.075) → 6 ajustes
  ≈ 8–11 h solo para esta pieza con 2 CPU. Sondas de 31 iteraciones (sin métricas): ctr 1 = 0.70 s/it (35 min por ajuste),
  ctr 2 con 12 categóricas 1.86 s/it, ctr 2 con CTR simplificados 2.09 s/it. Decisión: la rama normal de C2 usa
  **interacciones explícitas (apt_stand, apt_runway, apt_operator, route) + max_ctr_complexity 1**, 3000 it, lr 0.05, depth 8.
  El clasificador de compuerta (≈ 0.5 M filas, depth 6) mantiene ctr 2. Para medir lo que aporta ctr 2 en la rama normal se
  ejecuta una **ablación** (ctr 2, 2000 it, lr 0.075) en LOJO con semilla 42 (enero: el mes limpio), comparada en RMSE de
  clase 0. Si la ablación mejora la clase 0 en > 3 s, se documenta como pendiente de la etapa costosa.

## 10. Notas posteriores (operativas y diagnósticas; no cambian la decisión)

- **A2 (operativa).** La ablación ctr 2 de la rama normal murió dos veces por falta de memoria: la primera, lanzada a la vez
  que la etapa 2; la segunda, sola, hacia la iteración 550, con 6.1 GB de RSS frente al límite de ~5.8 GiB del entorno (las
  tablas de combinaciones de CTR crecen con los árboles). **No se ejecutó**: el efecto de ctr 2 en la rama normal queda sin
  medir y pendiente. La etapa 2, la selección y la decisión no dependen de ella.
- **Diagnósticos añadidos tras ver el resultado preliminar (no deciden):** sensibilidad sin las filas de clase 0 con y > 3 h
  (bin ya prefijado en §7) e influencia de filas individuales sobre las diferencias (Δ recalculado quitando cada una de las
  filas con mayor |e_a − e_b|). La regla de decisión (§8) se aplica sin cambios.
