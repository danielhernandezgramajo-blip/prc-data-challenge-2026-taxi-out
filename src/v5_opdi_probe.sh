#!/usr/bin/env bash
# V5 — Sondeo de viabilidad de OPDI (datos abiertos EUROCONTROL/OpenSky) en tu Mac (Terminal de macOS). NO sube nada.
#
#   cd ~/Desktop/Proyectos/Aviacion/"Aeropuertos Taxi time"
#   bash src/v5_opdi_probe.sh               # feb y ago 2025 (dos ventanas de 10 días): cobertura y calidad frente al target
#   bash src/v5_opdi_probe.sh --with-2026   # además, solo cobertura (sin target) en ene y jul 2026
#
# Descarga ficheros completos a data/external/opdi/raw/ (fuera de git), se queda solo con los vuelos de nuestros 10 aeropuertos
# (data/external/opdi/*_10apt.parquet) y borra los completos al terminar (usa --keep-raw para conservarlos).
# No toca los 14 Parquet originales ni las candidatas. Puede ejecutarse mientras corre v4b_run_candidate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-mac/bin/python ]; then
  echo "Falta el entorno .venv-mac: créalo primero con  bash src/v4b_run_candidate.sh  (o: python3 -m venv .venv-mac && .venv-mac/bin/pip install polars pyarrow numpy)" >&2
  exit 1
fi
mkdir -p outputs/experiments/v5
.venv-mac/bin/python src/v5_opdi_probe.py "$@" 2>&1 | tee outputs/experiments/v5/opdi_probe.log
echo "Listo. Resumen: outputs/experiments/v5/opdi_probe.txt — no se ha subido nada."
