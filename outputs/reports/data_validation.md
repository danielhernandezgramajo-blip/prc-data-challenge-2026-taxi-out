# Informe de validación de datos — PRC Data Challenge 2026

Generado: 2026-09-28 02:26 UTC · script: `src/validate_data.py`


## 1. Inventario de archivos

- ✅ 12 archivos training_2025-*.parquet (encontrados: 12)
- ✅ Los 12 nombres de training coinciden con los meses esperados
- ✅ ranking.parquet presente
- ✅ submitting.parquet presente
- ✅ Sin Parquet inesperados en Aeropuertos Taxi time/ (extra: [])

Carpeta de datos: `/sessions/rcw-019hjskbdqn1ka9xhlql86am/mnt/Aviacion/Aeropuertos Taxi time`


## 2–4. Apertura, esquema y dimensiones

- ✅ Todos los Parquet abren y se leen completos
- ✅ Columnas y tipos idénticos (nombre, orden y dtype) en los 12 training + ranking
- ✅ submitting.parquet tiene exactamente [MVT_ID_mvt, TAXITIME_SEC_mvt] con los mismos tipos que training
- ✅ Training: 4,167,797 filas (esperado 4,167,797)
- ✅ Training y ranking: 30 columnas
- ✅ Ranking: 689,534 filas (esperado 689,534)
- ✅ Submitting: 344,841 filas (esperado 344,841)
- ✅ Submitting: 2 columnas

| archivo | filas | cols | MB |
|---|---:|---:|---:|
| training_2025-01-01_2025-02-01.parquet | 307,257 | 30 | 21.2 |
| training_2025-02-01_2025-03-01.parquet | 287,411 | 30 | 19.8 |
| training_2025-03-01_2025-04-01.parquet | 328,622 | 30 | 22.6 |
| training_2025-04-01_2025-05-01.parquet | 350,288 | 30 | 23.8 |
| training_2025-05-01_2025-06-01.parquet | 370,105 | 30 | 25.2 |
| training_2025-06-01_2025-07-01.parquet | 365,986 | 30 | 25.0 |
| training_2025-07-01_2025-08-01.parquet | 381,161 | 30 | 26.0 |
| training_2025-08-01_2025-09-01.parquet | 382,047 | 30 | 26.0 |
| training_2025-09-01_2025-10-01.parquet | 367,679 | 30 | 25.1 |
| training_2025-10-01_2025-11-01.parquet | 371,163 | 30 | 25.4 |
| training_2025-11-01_2025-12-01.parquet | 324,530 | 30 | 22.2 |
| training_2025-12-01_2026-01-01.parquet | 331,548 | 30 | 22.7 |
| ranking.parquet | 689,534 | 30 | 43.6 |
| submitting.parquet | 344,841 | 2 | 1.7 |

## 5–6. Unicidad de MVT_ID_mvt y alineación ranking ↔ submitting

- ✅ MVT_ID_mvt sin nulos
- ✅ MVT_ID_mvt único en training (4,167,797 ids)
- ✅ MVT_ID_mvt único en ranking (689,534 ids)
- ✅ Sin solape de MVT_ID_mvt entre training y ranking (solape: 0)
- ✅ MVT_ID_mvt (float64) contiene únicamente valores enteros → puede convertirse a Int64 sin pérdida
- ✅ Nº filas DEP en ranking (344,841) == filas de submitting (344,841)
- ✅ IDs y ORDEN de submitting coinciden exactamente con las filas DEP de ranking
- ✅ Conjunto de IDs de submitting == conjunto de IDs DEP de ranking
- ✅ submitting.TAXITIME_SEC_mvt está completamente vacío (a rellenar)
- ✅ ranking DEP: TAXITIME_SEC_mvt es nulo (target oculto)
- ✅ ranking DEP: BLOCK_TIME_UTC_mvt es nulo (oculto: revelaría el target)
- ✅ ranking ARR: TAXITIME_SEC_mvt sí está informado (taxi-in disponible como contexto)

Distribución de PHASE_mvt:

| PHASE | training | ranking |
|---|---:|---:|
| ARR | 2,082,750 | 344,693 |
| DEP | 2,085,047 | 344,841 |

## 7. Comprobaciones adicionales (temporales y del target)

- ✅ Cada archivo training_YYYY-MM cubre ese mes según MVT_TIME_UTC_mvt (tolerancia 60 s en el límite inferior)

- ℹ️ 6 archivos contienen 1 fila con MVT_TIME_UTC_mvt unos segundos antes de la medianoche del mes (p. ej. 2025-02-28 23:59:57 en el archivo de marzo). Sin impacto práctico; los cortes temporales del backtesting se harán por MVT_TIME_UTC_mvt, no por archivo.
- ℹ️ Training: 1 fila(s) con TAXITIME_SEC_mvt nulo en total (todas ARR; ver siguiente comprobación).
- ✅ Training DEP: TAXITIME_SEC_mvt sin nulos (nulos en DEP: 0)
- ✅ Training DEP: TAXITIME_SEC_mvt == (MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt) en el 100 % de filas → el target es una función determinista de BLOCK_TIME (por eso está oculto en ranking)

- ℹ️ Training DEP: 369 taxi-out negativos y 19 iguales a 0 (se analizan en el EDA; no se eliminan aquí).
- ⚠️ Cadenas literales "NA" (no nulos reales) en columnas de texto: RUNWAY_mvt (186 train / 35 rank). Deben tratarse como ausentes en la preparación (`prepare_training.py`).
- ℹ️ Training cubre 2025-01-01 00:01:04+00:00 → 2025-12-31 23:57:36+00:00; ranking cubre 2026-01-01 00:01:55+00:00 → 2026-07-31 23:56:19+00:00 (el ranking es posterior al training: la validación debe ser temporal).

| mes | filas | filas DEP | negativos DEP | ceros DEP |
|---|---:|---:|---:|---:|
| 2025-01 | 307,257 | 153,706 | 46 | 2 |
| 2025-02 | 287,411 | 143,731 | 35 | 3 |
| 2025-03 | 328,622 | 164,450 | 26 | 2 |
| 2025-04 | 350,288 | 175,288 | 32 | 0 |
| 2025-05 | 370,105 | 185,202 | 21 | 0 |
| 2025-06 | 365,986 | 183,142 | 60 | 2 |
| 2025-07 | 381,161 | 190,713 | 34 | 1 |
| 2025-08 | 382,047 | 191,181 | 40 | 5 |
| 2025-09 | 367,679 | 183,951 | 16 | 2 |
| 2025-10 | 371,163 | 185,674 | 20 | 0 |
| 2025-11 | 324,530 | 162,332 | 18 | 0 |
| 2025-12 | 331,548 | 165,677 | 21 | 2 |

## Resultado

**TODAS LAS COMPROBACIONES PASAN.**