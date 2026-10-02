#!/usr/bin/env bash
# V5a — construye submissions/candidate_v5a.parquet en tu Mac (Terminal de macOS). NO sube nada al bucket.
#
#   cd ~/Desktop/Proyectos/Aviacion/"Aeropuertos Taxi time"
#   bash src/v5a_run_candidate.sh
#
# Requiere haber terminado antes src/v4b_run_candidate.sh (reutiliza sus predicciones por semilla de models/experiments/v4b/final/).
# Añade LightGBM al entorno .venv-mac si falta (en macOS LightGBM necesita libomp: si la importación falla, ejecuta
# `brew install libomp` y vuelve a lanzar). Entrena una rama normal LightGBM con los 12 meses (≈ 10–20 min) y la mezcla 50/50
# con la rama CatBoost de V4b. Validación: −1.41 s de RMSE_w frente a C2 en ene/jul 2025 (outputs/experiments/v5/outer_check.json).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="outputs/experiments/v5/v5a_run.log"
if [ ! -x .venv-mac/bin/python ]; then
  echo "Falta .venv-mac: ejecuta antes  bash src/v4b_run_candidate.sh" >&2; exit 1
fi
if [ -e submissions/candidate_v5a.parquet ]; then
  echo "submissions/candidate_v5a.parquet ya existe; no se sobrescribe." >&2; exit 1
fi
if ! .venv-mac/bin/python -c "import lightgbm" 2>/dev/null; then
  echo "== Instalando LightGBM en .venv-mac"
  .venv-mac/bin/pip install -q "lightgbm>=4.3"
fi
if ! .venv-mac/bin/python -c "import lightgbm; print('lightgbm', lightgbm.__version__)"; then
  echo "LightGBM no carga. En macOS suele faltar OpenMP: ejecuta  brew install libomp  y vuelve a lanzar este script." >&2
  exit 1
fi
echo "== Pruebas rápidas de V5"
.venv-mac/bin/python -m pytest -q tests/test_v5.py
mkdir -p outputs/experiments/v5
echo "== Entrenamiento LightGBM y mezcla — registro en $LOG"
.venv-mac/bin/python src/v5a_final.py 2>&1 | tee "$LOG"
echo "== Resultado"
shasum -a 256 submissions/candidate_v5a.parquet
echo "Listo. No se ha subido nada. Revisa submissions/candidate_v5a.json antes de decidir la subida."
