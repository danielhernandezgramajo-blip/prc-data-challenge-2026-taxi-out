"""
Ejecuta todo el EDA sin notebook (útil para CI o máquinas sin Jupyter).
Genera exactamente las mismas tablas y figuras que `notebooks/01_eda.ipynb`.

Uso: python src/run_eda.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda  # noqa: E402


def main() -> None:
    t0 = time.time()
    steps = [("diccionario", eda.data_dictionary), ("top_values", eda.top_values)]
    for name, fn in steps:
        fn(); print(f"[{time.time()-t0:6.1f}s] {name}")
    tr, rk = eda.load_dep()
    print(f"[{time.time()-t0:6.1f}s] carga DEP: train={tr.height:,} rank={rk.height:,}")
    for name, fn in [("target", lambda: eda.target_distribution(tr)), ("outliers", lambda: eda.outliers(tr)),
                     ("airport", lambda: eda.by_airport(tr)), ("calendar", lambda: eda.by_calendar(tr)),
                     ("operational", lambda: eda.by_operational(tr)), ("train_vs_rank", lambda: eda.train_vs_ranking(tr, rk)),
                     ("leakage", lambda: eda.leakage_and_redundancy(tr, rk)), ("congestion", lambda: eda.congestion_signal(tr))]:
        fn(); print(f"[{time.time()-t0:6.1f}s] {name}")
    print("EDA completo.")


if __name__ == "__main__":
    main()
