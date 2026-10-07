"""
Pruebas V7: congestión de vecinos (sin la propia fila, solo A en (0, 2 h), mismo aeropuerto y ventana), máscara de V7c
(solo filas sin AOBT_3, fuera de la celda, sin regla LOBT, nba > umbral) y max(b, nb); scripts sin comandos de subida.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def test_neighbour_congestion_excludes_self_and_other_airports():
    from v7_congestion import neighbour_congestion
    t0 = datetime(2026, 1, 3, 10, 0, 0)
    df = pl.DataFrame({
        "MVT_TIME_UTC_mvt": [t0, t0 + timedelta(minutes=10), t0 + timedelta(minutes=20), t0 + timedelta(minutes=200), t0],
        "ADEP_mvt": ["EHAM", "EHAM", "EHAM", "EHAM", "LFPG"],
        "mvt_minus_aobt3": [1000.0, 2000.0, 9000.0, 3000.0, 500.0],   # 9000 s (> 2 h) no cuenta como vecino válido
    }).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone("UTC"))
    nb = neighbour_congestion(df, window_min=60)
    assert np.isclose(nb[0], 2000.0)          # vecinos válidos de la fila 0: solo la fila 1
    assert np.isclose(nb[1], 1000.0)          # la fila 2 tiene A > 2 h → excluida
    assert np.isclose(nb[2], 1500.0)          # media de filas 0 y 1
    assert np.isnan(nb[3])                    # a 200 min: sin vecinos en ±60 min
    assert np.isnan(nb[4])                    # otro aeropuerto: no se mezcla


def test_noaobt_mask_and_max_nb():
    from v7c_noaobt import apply_max_nb, noaobt_mask
    nba = np.array([700.0, 700.0, 700.0, 700.0, 500.0, np.nan])
    has = np.array([0, 1, 0, 0, 0, 0])
    L = np.array([np.nan, np.nan, 8000.0, np.nan, np.nan, np.nan])
    cell = np.array([False, False, False, True, False, False])
    nb = np.array([2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0])
    m = noaobt_mask(nba, has, L, cell, nb, 600.0)
    assert m.tolist() == [True, False, False, False, False, False]   # con AOBT_3, regla LOBT, celda, nba bajo o NaN → fuera
    b = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    out = apply_max_nb(b, nb, m)
    assert out[0] == 2500.0 and (out[1:] == 1000.0).all()
    assert apply_max_nb(np.array([3000.0]), np.array([2500.0]), np.array([True]))[0] == 3000.0   # nunca baja


def test_v7_scripts_do_not_upload():
    for name in ("v7_final.py", "v7c_noaobt.py", "v7_congestion.py", "v7b_neighbour.py"):
        p = ROOT / "src" / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        for bad in ("boto3", "aws s3", "s3 cp", "upload_file", "put_object", "git push", "mc cp"):
            assert bad not in txt, (name, bad)
