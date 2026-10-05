#!/usr/bin/env bash
# V6 — reconstruye submissions/candidate_v6.parquet en tu Mac (Terminal de macOS). NO sube nada al bucket.
#
#   cd ~/Desktop/Proyectos/Aviacion/"Aeropuertos Taxi time"
#   bash src/v6_run_candidate.sh
#
# Solo hace falta si quieres reproducir la candidata: candidate_v6.parquet ya viene calculada (en la nube, con las mismas
# features —verificadas por hash— y tus predicciones por semilla de V4b). Requiere models/experiments/v4b/final/*__rank.npy
# y la caché LightGBM models/experiments/v5/final/lgb_const__s42__rank.npy (si falta, ejecuta antes src/v5a_run_candidate.sh).
# Tarda segundos: no entrena nada, solo ajusta la recta de la regla LOBT con 2025 y mezcla.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="outputs/experiments/v6/v6_run.log"
if [ ! -x .venv-mac/bin/python ]; then
  echo "Falta .venv-mac: ejecuta antes  bash src/v4b_run_candidate.sh" >&2; exit 1
fi
if [ -e submissions/candidate_v6.parquet ]; then
  echo "submissions/candidate_v6.parquet ya existe; no se sobrescribe (usa:  .venv-mac/bin/python src/v6_final.py --out /tmp/v6_check.parquet)." >&2
  exit 1
fi
echo "== Pruebas rápidas de V6"
.venv-mac/bin/python -m pytest -q tests/test_v6.py
mkdir -p outputs/experiments/v6
echo "== Candidata V6 — registro en $LOG"
.venv-mac/bin/python src/v6_final.py 2>&1 | tee "$LOG"
echo "== Resultado"
shasum -a 256 submissions/candidate_v6.parquet
echo "Listo. No se ha subido nada. Revisa submissions/candidate_v6.json antes de decidir la subida."
