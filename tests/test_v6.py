"""
Pruebas V6: ajuste y aplicación de la regla de hueco LOBT (solo filas con MVT − LOBT > 2 h, nulos intactos, recorte a 0),
activación de la compuerta por prefijo según su comprobación prerregistrada, selección de filas y prefijos de la compuerta,
y ausencia de comandos de subida en los scripts V6.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def _rule_frame(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    L = np.concatenate([rng.uniform(7300, 15000, 200), rng.uniform(0, 7000, 300), [np.nan] * 20])
    y = np.where(L > 7200, -1200 + 1.02 * np.nan_to_num(L) + rng.normal(0, 50, L.size), rng.uniform(300, 1500, L.size))
    y[:5] = 0                                    # filas con y = 0 dentro del patrón: no deben entrar en el ajuste
    return pl.DataFrame({"mvt_minus_lobt": L, "TAXITIME_SEC_mvt": np.round(y).astype(np.int32)})


def test_fit_lobt_rule_recovers_line_and_ignores_other_rows():
    from v6_final import LOBT_GAP_SEC, fit_lobt_rule
    r = fit_lobt_rule(_rule_frame())
    assert r["threshold_sec"] == LOBT_GAP_SEC == 7200.0
    assert r["n"] == 195                          # 200 del patrón − 5 con y = 0; ni las ≤ 2 h ni los nulos
    assert abs(r["c"] - 1.02) < 0.01 and abs(r["a"] + 1200) < 120


def test_apply_lobt_rule_only_touches_long_gaps_and_clips():
    from v6_final import apply_lobt_rule
    rule = {"a": -1228.2, "c": 1.0153, "threshold_sec": 7200.0}
    pred = np.array([500.0, 600.0, 700.0, 800.0, 900.0])
    L = np.array([7200.0, 7201.0, np.nan, 86832.0, 100.0])
    out = apply_lobt_rule(pred, L, rule)
    assert out[0] == 500.0 and out[2] == 700.0 and out[4] == 900.0       # umbral estricto, NaN y huecos cortos intactos
    assert np.isclose(out[1], -1228.2 + 1.0153 * 7201.0)
    assert np.isclose(out[3], -1228.2 + 1.0153 * 86832.0)
    neg = apply_lobt_rule(np.array([10.0]), np.array([8000.0]), {"a": -9000.0, "c": 1.0, "threshold_sec": 7200.0})
    assert neg[0] == 0.0 and (pred == np.array([500.0, 600.0, 700.0, 800.0, 900.0])).all()   # no muta la entrada


def test_prefix_gate_enabled_follows_check(tmp_path, monkeypatch):
    import v6_final
    assert v6_final.prefix_gate_enabled("off")[0] is False
    assert v6_final.prefix_gate_enabled("on")[0] is True
    p = tmp_path / "chk.json"
    monkeypatch.setattr(v6_final, "PREFIX_CHECK", p)
    assert v6_final.prefix_gate_enabled("auto")[0] is False                  # sin comprobación → desactivada
    p.write_text(json.dumps({"pass": False, "delta_RMSE_w": 0.42}), encoding="utf-8")
    assert v6_final.prefix_gate_enabled("auto")[0] is False
    p.write_text(json.dumps({"pass": True, "delta_RMSE_w": -1.5}), encoding="utf-8")
    assert v6_final.prefix_gate_enabled("auto")[0] is True


def test_prefix_rows_and_prefix_extraction():
    from v6_prefix_gate import prefix_expr, target_rows_expr
    df = pl.DataFrame({
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LIRF", "EHAM", "LIRF"],
        "FLIGHT_mvt": ["RYR975", "KL1234", "TAP83RL", "ISR344", "RYR1", None],
        "mvt_minus_sched": [4000.0, 4000.0, 1000.0, 4000.0, 4000.0, 4000.0],
        "mvt_minus_eobt1": [None, None, None, 50.0, None, None],
        "mvt_minus_lobt": [None, None, None, 60.0, None, None],
        "has_aobt3": [1, 0, 0, 1, 0, 0],
        "days_after_sched": [0, 0, 0, 0, 0, 1],
        "MVT_TIME_UTC_mvt": [datetime(2025, 1, 1)] * 6,
    }).with_columns(target_rows_expr(), prefix_expr())
    # fila 0: LIRF sin NM, s > 1800 → sí; 1: sí; 2: s ≤ 1800 → no; 3: tiene NM → no; 4: otro aeropuerto → no;
    # 5: celda de fecha desplazada sin AOBT_3 → no
    assert df["pg_rows"].to_list() == [True, True, False, False, False, False]
    assert df["prefix"].to_list()[:4] == ["RYR", "KL", "TAP", "ISR"]


def test_v6_scripts_do_not_upload():
    for name in ("v6_final.py", "v6_prefix_gate.py", "v6_run_candidate.sh"):
        p = ROOT / "src" / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        for bad in ("boto3", "aws s3", "s3 cp", "upload_file", "put_object", "git push", "mc cp"):
            assert bad not in txt, (name, bad)
