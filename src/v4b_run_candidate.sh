#!/usr/bin/env bash
# V4b — construye submissions/candidate_v4b.parquet en tu Mac (Terminal de macOS). NO sube nada al bucket.
#
#   cd ~/Desktop/Proyectos/Aviacion/"Aeropuertos Taxi time"
#   bash src/v4b_run_candidate.sh              # 3 semillas, como en la validación
#   SEEDS="42" bash src/v4b_run_candidate.sh   # 1 semilla: ~3 veces más rápido, algo menos estable
#
# Crea (una sola vez) un entorno .venv-mac con lo mínimo necesario, ejecuta las pruebas rápidas de V4b y entrena.
# Si se interrumpe, al relanzarlo reutiliza lo ya entrenado (models/experiments/v4b/final/).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-42 7 2025}"
PY="${PY:-python3}"
LOG="outputs/experiments/v4b/final_run.log"

if [ ! -x .venv-mac/bin/python ]; then
  echo "== Creando entorno .venv-mac con $($PY --version)"
  "$PY" -m venv .venv-mac
  .venv-mac/bin/pip install -q --upgrade pip
  .venv-mac/bin/pip install -q "polars>=1.0" "pyarrow>=15" "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" "scipy>=1.10" "catboost>=1.2" "pytest>=7"
fi
.venv-mac/bin/python -c "import catboost, polars, sklearn, scipy, os; print('catboost', catboost.__version__, '| polars', polars.__version__, '| hilos', os.cpu_count())"

if [ -e submissions/candidate_v4b.parquet ]; then
  echo "submissions/candidate_v4b.parquet ya existe; no se sobrescribe. Muévela o bórrala si quieres regenerarla." >&2
  exit 1
fi

echo "== Pruebas rápidas de V4b"
.venv-mac/bin/python -m pytest -q tests/test_v4b.py -k "not rank_features"

echo "== Entrenamiento y predicción (semillas: $SEEDS) — registro en $LOG"
mkdir -p outputs/experiments/v4b
# shellcheck disable=SC2086
.venv-mac/bin/python src/v4b_final.py --seeds $SEEDS 2>&1 | tee "$LOG"

echo "== Resultado"
shasum -a 256 submissions/candidate_v4b.parquet
echo "Listo. No se ha subido nada. Revisa submissions/candidate_v4b.json antes de decidir la subida."
