# Diagnóstico tras el leaderboard (2026-10-01): dónde están los 98 s que nos separan del líder

Resultado: `optimistic-dragon_v1` (candidate_v3, S4A_mix) → RMSE **317.99 s**, puesto 125. Líder 219.6; top-10 ≤ 233; un equipo
con solo 8 envíos está en 230.0. Este documento no usa el leaderboard para ajustar nada (política en `protocol_amendments.md`):
usa el puesto como motivo para preguntar a los datos de 2025 dónde está el hueco. Todos los números siguientes son de 2025.

## 1. Descomposición del error de S4A_mix en el backtest de julio 2025 (el test es 56 % julio)

| Grupo | filas | % del MSE | RMSE en el grupo |
|---|---:|---:|---:|
| clase 0, taxi ≤ 20 min | 71.7 % | 27.9 % | 219 |
| clase 0, 20–60 min | 23.9 % | 30.0 % | 393 |
| clase 0, 1–3 h | 0.19 % | 11.2 % | 2 668 |
| clase 0, > 3 h | 3 filas | 1.1 % | 9 183 |
| clase 1 (BLOCK = SCHED) | 4.2 % | 20.5 % | 776 |
| clase 2 (fecha desplazada) | 4 filas | 9.3 % | 23 377 |

Cotas (no modelos): clasificador de régimen perfecto → julio 350.7 → **293.7**; además cola de clase 0 perfecta → 266.8.
Solo el grueso (clase 0 ≤ 1 h) con la mezcla S4A: 272.9 (la masa p1 de la mezcla cuesta ~30 s en el grueso; un modelo
directo solo-clase-0 da 241.9 en ese mismo grueso). Diciembre: total 234.4; grueso 214.

## 2. Hipótesis probadas para identificar el artefacto BLOCK = SCHED de forma (casi) determinista — NINGUNA funciona

| Hipótesis (todas con columnas visibles en ranking) | Resultado (2025) |
|---|---|
| Caídas del sistema del aeropuerto: tasa de artefacto en las LLEGADAS del mismo aeropuerto a ±15/30/60/180 min | AUC 0.56 / 0.60 / 0.63 / 0.66 |
| Agrupamiento temporal puro (otras salidas ±30 min; oráculo, no disponible) | AUC 0.68 |
| Feed de la aerolínea: llegada del MISMO vuelo (15 % de salidas) con artefacto | tasa 4.5 % vs 4.2 %: nada |
| Tasa de artefacto de llegadas por operador×mes / operador×aeropuerto×mes | AUC 0.64 / 0.67 |
| Mismo avión (llegada previa en el mismo stand, 98 % enlazables, 83–88 % mismo tipo/operador) | 14.3 % vs 4.6 %; AUC 0.53 |
| Mismo número de vuelo en el mes (leave-one-out) | AUC 0.68 |
| Identificadores (MVT_ID, FLIGHT_ID, paridad, desorden temporal), resolución de segundos, discrepancias mvt/flt | AUC ≤ 0.51 |
| Otras fuentes de relleno (EOBT_1 / LOBT / IOBT / AOBT_3) para las filas 1–3 h "normales" | 44 / 23 / 22 / 4 de 371 en julio: marginal |
| Clasificador actual (features V1 + fecha desplazada) | AUC 0.87, bien calibrado |

Conclusión: con estas columnas el artefacto no es determinista; parece aleatorio dado lo observable. El techo realista del
clasificador está poco por encima del actual.

## 3. Grueso: ¿capacidad o información?

Experimento `exp_capacity.py` (entrena ene–jun clase 0, valida julio clase 0 ≤ 1 h):

| Config | RMSE grueso julio | MAE | 20–60 min |
|---|---:|---:|---:|
| A actual (1000 it, depth 6) | 241.9 | 164.9 | 368 |
| B 3000 it, depth 8, lr 0.05 | 237.3 | 161.1 | 362 |
| C = B + escala (llegada previa en el stand: min desde aterrizaje, retraso on-block, taxi-in) + FLIGHT_mvt | **232.6** | 158.6 | 352 |

Otras features contemporáneas probadas sobre el residuo del grueso: cola de pista anclada en AOBT_3 (salidas/llegadas entre
AOBT_3 y mi despegue), mediana/p90 del taxi-in de llegadas ±30/60 min, e incluso el oráculo del taxi-out real de las otras
salidas ±30 min: ninguna reduce el residuo más de 2 s. Causa: BLOCK_TIME (aeropuerto) y AOBT_3 (NM) difieren ±5 min sin
estructura visible (solo 21 % dentro de ±60 s; la dispersión dentro de operador×aeropuerto×mes es 291 s, entre grupos 142 s).
Cuando AOBT_3 es exacta, MVT−AOBT_3 daría 39 s y el modelo da 148, pero no hay forma de saber cuándo lo es.

## 4. Qué es alcanzable con los datos

- Etapa 3 (grueso: capacidad + escala + nº de vuelo; clasificador con las mismas features nuevas): estimación −10 a −20 s
  en el test → ~300. Puesto estimado ~100–110. Coste: ~3 h de cómputo bajo el protocolo de 8 folds.
- Para top-10 (≤ 233) hace falta eliminar casi por completo el error de clases 1/2 y de la cola de clase 0, es decir,
  conocer por fila lo que los datos no contienen. Con 60–90 envíos se pueden sondear individualmente las ~100 filas
  más extremas (cada envío revela un escalar). No lo hemos hecho ni lo haremos sin decisión explícita: contradice la
  política de `protocol_amendments.md` y, según las reglas, puede ser ilegítimo. Un equipo con 8 envíos a 230 sugiere,
  no obstante, que algo modelable se nos escapa; no lo hemos encontrado en 2025.
