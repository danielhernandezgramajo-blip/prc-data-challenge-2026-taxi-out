"""
Pruebas V5: codificación de la media del target fuera de mes (sin fuga del propio mes, ni en el prior), rama normal LightGBM
en datos sintéticos (hojas constantes y lineales), mezcla y recorte de la candidata V5a, ausencia de comandos de subida,
utilidades del sondeo OPDI (zonas horarias y rejilla de 10 días).
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def _frame(n_per_month: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, 13):
        for i in range(n_per_month):
            rows.append({"MVT_TIME_UTC_mvt": datetime(2025, m, 1 + i % 27, 10, 0, 0), "cat": f"c{rng.integers(0, 5)}"})
    return pl.DataFrame(rows)


def test_out_of_month_te_excludes_own_month():
    from v5_lgb import OutOfMonthTE
    df = _frame()
    y = np.zeros(df.height)
    months = df["MVT_TIME_UTC_mvt"].dt.month().to_numpy()
    y[months == 3] = 1000.0                       # solo marzo tiene target alto
    te = OutOfMonthTE(["cat"], k=20.0).fit_transform(df, y)[:, 0]
    # las filas de marzo no pueden ver su propio target: su codificación (y su prior) sale de los otros 11 meses → 0
    assert np.allclose(te[months == 3], 0.0)
    # las de otros meses sí ven marzo (a través de la categoría y del prior)
    assert (te[months != 3] > 0).all()


def test_out_of_month_te_transform_uses_all_months_and_unknown_to_prior():
    from v5_lgb import OutOfMonthTE
    df = _frame()
    y = np.arange(df.height, dtype=float)
    enc = OutOfMonthTE(["cat"], k=0.0)
    enc.fit_transform(df, y)
    new = pl.DataFrame({"cat": ["c0", "zzz"], "MVT_TIME_UTC_mvt": [datetime(2026, 1, 5), datetime(2026, 1, 5)]})
    out = enc.transform(new)[:, 0]
    assert np.isclose(out[0], y[(df["cat"] == "c0").to_numpy()].mean())
    assert np.isclose(out[1], y.mean())


def test_lgb_normal_smoke_const_and_linear(monkeypatch):
    import v5_lgb
    from config import TARGET
    rng = np.random.default_rng(1)
    n = 3000
    num = ["x1", "x2"]
    monkeypatch.setattr(v5_lgb, "num_cols", lambda: num)
    monkeypatch.setattr(v5_lgb, "te_cols", lambda: ["hc"])
    monkeypatch.setattr(v5_lgb, "NATIVE_CAT", ["lc"])
    x1 = rng.uniform(0, 1000, n)
    df = pl.DataFrame({"MVT_TIME_UTC_mvt": [datetime(2025, 1 + i % 12, 1 + i % 27, 8) for i in range(n)],
                       "x1": x1, "x2": rng.normal(size=n), "hc": [f"h{i % 50}" for i in range(n)], "lc": [f"l{i % 3}" for i in range(n)],
                       "mvt_minus_sched": np.full(n, 5000.0), TARGET: (300 + x1 + rng.normal(0, 5, n)).round().astype(np.int32)})
    for kind in ("const", "linear"):
        m = v5_lgb.LGBNormal(0, kind, iterations=30, params={"min_data_in_leaf": 20, "num_leaves": 15})
        m.num, m.native, m.te = num, ["lc"], ["hc"]
        m.fit(df)
        p = m.predict(df)
        assert p.shape == (n,) and np.isfinite(p).all()
        rmse = float(np.sqrt(np.mean((p - df[TARGET].to_numpy()) ** 2)))
        assert rmse < 200, (kind, rmse)


def test_v5a_combine_blend_and_clip():
    from v4b_final import combine
    n = 4
    P0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]], dtype=float)
    c2n = np.array([600.0, 600, 600, 600]); lgb = np.array([800.0, 800, 800, 800])
    b = 0.5 * c2n + 0.5 * lgb
    s = np.array([5000.0, 5000, 5000, -9000]); p1 = np.array([0.0, 0.5, 1.0, 0.5])
    cell = np.array([False, True, True, False])
    out = combine(P0, b, p1, s, cell)
    assert np.allclose(out[:3], [700.0, 5000.0, 86400 + 700.0])
    assert out[3] == 0.0 and (out >= 0).all() and n == out.size


def test_v5_scripts_do_not_upload():
    for name in ("v5a_final.py", "v5_opdi_probe.py", "v5_lgb.py", "v5_gate.py", "v5_opdi_probe.sh", "v5a_run_candidate.sh"):
        p = ROOT / "src" / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        for bad in ("boto3", "aws s3", "s3 cp", "upload_file", "put_object", "git push"):
            assert bad not in txt, (name, bad)


def test_opdi_naive_utc_and_grid():
    from v5_opdi_probe import PROBE_2025, PROBE_2026, grid_start, naive_utc
    for d0 in list(PROBE_2025.values()) + list(PROBE_2026.values()):
        assert grid_start(d0) == d0
    assert grid_start(date(2022, 1, 1)) == date(2022, 1, 1) and grid_start(date(2022, 1, 15)) == date(2022, 1, 11)
    df = pl.DataFrame({"a": [datetime(2025, 2, 4, 10, 0, 0)]}).with_columns(pl.col("a").dt.replace_time_zone("UTC"))
    df = df.with_columns(pl.col("a").alias("b"), pl.lit(1738663200).alias("c"), pl.lit(1738663200000).alias("d"))
    for c in ("a", "b", "c", "d"):
        out = naive_utc(df, c)[c][0]
        assert out == datetime(2025, 2, 4, 10, 0, 0), (c, out)
