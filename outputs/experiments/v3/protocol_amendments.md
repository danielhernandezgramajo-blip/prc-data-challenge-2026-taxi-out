# Registro formal de enmiendas al protocolo (V2 → V3)

El protocolo original (prompt V2) fijaba: 8 folds de origen creciente, diciembre como holdout abierto una sola vez con la
finalista congelada, criterios c1–c8 y bootstrap, y "no cambiar silenciosamente folds, métricas o población". Este registro
enumera **todas** las desviaciones, cuándo se decidieron respecto a los resultados que ya se habían visto, y qué decisión
habría cambiado sin ellas. Nada de lo que sigue se hizo en silencio, pero hasta hoy no estaba reunido en un solo sitio.

| # | Fecha | Enmienda | Motivo | Decidida… | Efecto sobre las decisiones |
|---|---|---|---|---|---|
| E1 | 2026-09-29 (V2) | **c8** ("la ganancia no depende de pocas filas": < 90 % de la reducción de SSE en las 50 mejores filas) pasa de obligatorio a **diagnóstico** | S_mix corrige un régimen raro (bloque = hora programada); por construcción su ganancia está en pocas filas | **después** de ver que S_mix incumplía c8 en desarrollo (share = 0,94); antes de abrir diciembre | Sin E1, S_mix no habría sido finalista y `candidate_v2` no existiría. Documentado en `outputs/experiments/v2/report.md` §8 |
| E2 | 2026-09-29 (V3) | Diciembre se abre por **segunda** vez, con la finalista de la etapa 2 congelada (S2A_mix) | La etapa 2 no estaba prevista en el protocolo original, que contemplaba una sola apertura | antes de ver diciembre para las candidatas de la etapa 2 | Diciembre deja de ser un holdout limpio para toda la etapa 2; la evidencia limpia de la etapa 2 son los 7 folds de desarrollo |
| E3 | 2026-09-29 | Se define **S3A_mix** (banda de retraso con umbral fijo de 12 h) | El diagnóstico de diciembre mostró que la celda plana reparte masa de clase 2 a filas de retraso corto | **después** de ver diciembre y la tabla cruzada de todo 2025 | Elegida como `final_choice` con etiqueta post-hoc; la regla prefijada seguía diciendo S2A_mix |
| E4 | 2026-09-30 | **S4A_mix**: el umbral (6/9/12/18/24 h o celda plana) se elige por log-loss leave-one-out **dentro del bloque de entrenamiento** de cada fold | Eliminar la elección manual del parámetro (revisión externa) | después de E3; el parámetro ya no usa diciembre, la **hipótesis** sigue siendo posterior a diciembre | Sustituye a S3A como `final_choice`; con todo 2025 elige 12 h → `candidate_v3.parquet` byte-idéntica |
| E5 | 2026-09-30 | **c8 redefinido** como `c8_no_harm_bulk` (RMSE de la población normal no empeora > 1 s y ≤ 2 de 10 aeropuertos empeoran); el c8 original se reporta como `c8_old_share_top50` | Revisión externa: el c8 original es inaplicable a correcciones de régimen raro | **después** de ver los resultados de la etapa 2 | No cambia la finalista de la regla prefijada (S2A_mix); S2_mix (solo celdas planas) deja de cumplir |

## Decisión formal sobre c8

La métrica del concurso es el RMSE. En estos datos el error cuadrático lo dominan unas decenas de filas con errores de
miles de segundos (en desarrollo, 10 filas de clase 2 concentraban el 25 % del MSE restante de S_mix; en el ranking, una
sola fila con 60 000 s de error mueve el RMSE final ≈ 20 s). Un criterio que prohíba ganancias concentradas en pocas filas
prohíbe, por construcción, las únicas ganancias grandes disponibles, y por tanto contradice el objetivo. Por eso:

1. La **concentración de la ganancia NO es un veto**. Es un **diagnóstico de fragilidad** que se reporta siempre, con tres
   medidas: `share_gain_top50_rows`, la ganancia **excluyendo la celda LIRF** (`gain_decomposition_lirf_cell.json`) y la
   estabilidad del signo por fold (7/7) y por aeropuerto (10/10).
2. La condición obligatoria que sustituye a c8 es **no dañar al grueso** (`c8_no_harm_bulk`).
3. Toda candidata cuya ganancia esté concentrada se etiqueta como **apuesta**, con la cuantía de la apuesta separada de la
   parte robusta (tabla siguiente).

## Descomposición de la ganancia: apuesta (celda LIRF) frente a parte robusta

| RMSE (s) frente a S_mix | Total | Fuera de la celda LIRF (parte robusta) | Dentro de la celda (apuesta) | Filas de la celda |
|---|---:|---:|---:|---:|
| Desarrollo (may–nov, 1 282 195 filas): S2A_mix | −20,0 | −2,4 | −17,6 | 50 |
| Desarrollo: S4A_mix (`candidate_v3`) | −23,1 | −2,4 | −20,7 | 50 |
| Diciembre (165 677 filas): S2A_mix | −3,3 | −4,1 | +0,8 | 5 |
| Diciembre: S4A_mix | −6,7 | −4,1 | −2,6 | 5 |

Lectura: la parte robusta de la etapa 2 (features de congestión en la rama normal) vale 2–4 s y mejora los 10 aeropuertos;
todo lo demás es la apuesta sobre las filas de LIRF con fecha desplazada (25 en el ranking). Si el artefacto no se
comporta en 2026 como en 2025, se pierde la apuesta y la parte robusta no la compensa.

## Escalera de riesgo (clasificación acordada con la revisión externa)

| Candidata | Naturaleza | Depende de que el artefacto BLOCK = SCHED persista en 2026 |
|---|---|---|
| V1 (`candidate_v1`) | control conservador | implícitamente sí: se entrenó con los mismos targets contaminados, pero no apuesta de forma explícita |
| V2 S_mix (`candidate_v2`) | apuesta moderada, confirmada en la primera apertura de diciembre | sí (explícitamente, en ~100 k filas de clase 1) |
| S2A_mix | ampliación preseleccionada de la etapa 2 (regla prefijada); mejora rigurosamente defendible | sí, + celdas planas |
| S3A_mix / S4A_mix (`candidate_v3`) | apuesta más agresiva y post-hoc en ~25 filas del ranking | sí, + bandas de retraso |

Que el patrón exista en las variables del ranking 2026 (misma tasa BLOCK = SCHED en las llegadas, 25 filas en la celda)
demuestra que la regla es aplicable, no que conserve la misma relación con el target. Solo el ranking privado lo dirá.

## Política de subida y de leaderboard (prefijada ahora, antes de subir nada)

1. Como máximo **dos** submissions decididas a priori: `candidate_v2` (apuesta moderada) y `candidate_v3` (apuesta agresiva).
   Si solo hay una oportunidad: `candidate_v3` (mayor valor esperado).
2. El resultado del leaderboard se registra **una vez** en `decision.json` y no se usa para seleccionar ni ajustar reglas,
   umbrales, α ni pesos. Si v3 queda por detrás de v2, la única acción permitida es volver a la candidata preseleccionada
   (S2A_mix o V2), no re-ajustar sobre el leaderboard.
3. Cualquier trabajo posterior se evalúa con el mismo protocolo de 8 folds; diciembre no vuelve a abrirse.
