"""
Pruebas V4b: particiones LOJO/LOJuO con meses internos, features sin target (y mismas columnas en training y ranking),
ventanas de llegadas, piezas de la etapa 2 (umbral duro, calibraciones, mezcla de celda, recorte), ausencia de fuga en la
validación cruzada interna, selección anidada, bootstrap ponderado y coherencia de las salidas. La reproducibilidad de una
candidata no aplica: V4b no genera candidata (se comprueba que no existe).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
V4B = ROOT / "outputs" / "experiments" / "v4b"


def _year_frame(n_per_month: int = 50) -> pl.DataFrame:
    rows = []
    for m in range(1, 13):
        for i in range(n_per_month):
            rows.append({"MVT_ID_mvt": m * 1000 + i, "MVT_TIME_UTC_mvt": datetime(2025, m, 1 + i % 27, 10, 0, 0)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Particiones: 9 meses de entrenamiento, internos feb/ago, outer = mes evaluado; sin solapes
# ---------------------------------------------------------------------------
def test_split_folds_disjoint():
    from v4b_protocol import INNER, OUTER, split_fold, train_months
    df = _year_frame()
    for fold, ev in OUTER.items():
        tr, inn, out = split_fold(df, fold)
        mt = set(tr["MVT_TIME_UTC_mvt"].dt.month().unique().to_list())
        assert mt == set(train_months(fold)) and len(mt) == 9 and ev not in mt and not mt & set(INNER)
        assert set(inn["MVT_TIME_UTC_mvt"].dt.month().unique().to_list()) == set(INNER)
        assert set(out["MVT_TIME_UTC_mvt"].dt.month().unique().to_list()) == {ev}
        ids = [set(x["MVT_ID_mvt"].to_list()) for x in (tr, inn, out)]
        assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
        assert sum(len(i) for i in ids) == df.height
    assert OUTER == {"LOJO": 1, "LOJuO": 7} and INNER == (2, 8)


# ---------------------------------------------------------------------------
# 2. Features: nunca BLOCK/TAXITIME de las salidas; ventanas de llegadas correctas
# ---------------------------------------------------------------------------
def test_feature_inputs_target_free():
    import v4b_features as F
    assert "BLOCK_TIME_UTC_mvt" not in F.DEP_RAW and "TAXITIME_SEC_mvt" not in F.DEP_RAW
    for c in F.num_cols() + F.cat_cols():
        assert c not in ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt")
        assert not c.startswith("ARVT")            # ARVT fuera de la variante primaria


def test_window_rate():
    from v4b_features import window_rate
    te = np.array([0.0, 100.0, 200.0, 5000.0]); flag = np.array([1.0, 0.0, 1.0, 1.0])
    n, r = window_rate(np.array([100.0, 3000.0, 9000.0]), te, flag, 150.0)
    assert n.tolist() == [3.0, 0.0, 0.0] and np.isclose(r[0], 2 / 3) and np.isnan(r[1]) and np.isnan(r[2])
    n2, r2 = window_rate(np.array([4900.0]), te, flag, 100.0)
    assert n2[0] == 1 and r2[0] == 1.0


def test_rank_features_match_train_schema():
    from v4b_features import build_v4b, cat_cols, features_path, num_cols
    if not (ROOT / "ranking.parquet").exists():
        pytest.skip("sin ranking.parquet")
    rk = build_v4b("rank")
    assert rk.height == 344_841
    needed = [c for c in num_cols() + cat_cols() if c not in ("rel_dep_pm30", "rel_dep_hour")]
    assert all(c in rk.columns for c in needed)
    assert rk["TAXITIME_SEC_mvt"].null_count() == rk.height        # el target de ranking está oculto y no se usa
    if features_path("train").exists():
        tr = pl.read_parquet(features_path("train"), n_rows=10)
        assert tr.columns == rk.columns
        assert all(tr[c].dtype == rk[c].dtype or tr[c].dtype == pl.Null or rk[c].dtype == pl.Null for c in needed)


# ---------------------------------------------------------------------------
# 3. Etapa 2: piezas puras
# ---------------------------------------------------------------------------
def test_hard_threshold_matches_bruteforce():
    from v4b_stage2 import fit_hard_threshold
    rng = np.random.default_rng(3)
    n = 300
    p = rng.random(n); s = rng.normal(5000, 3000, n); b = rng.normal(900, 200, n)
    y = np.where(rng.random(n) < p ** 3, s, b + rng.normal(0, 150, n))
    t = fit_hard_threshold(p, y, b, s)
    sse = lambda th: (((np.where(p > th, s, b)) - y) ** 2).sum()
    best = min([sse(th) for th in np.concatenate([[1.01], np.sort(p) - 1e-9])])
    assert np.isclose(sse(t), best)


def test_rmse_calibration_and_platt():
    from v4b_stage2 import apply_calibration, fit_platt, fit_rmse_calibration, sigmoid, logit
    rng = np.random.default_rng(5)
    n = 20_000
    z = rng.normal(-2, 1.5, n); q_true = sigmoid(1.7 * z + 0.4)
    y1 = rng.random(n) < q_true
    p_raw = sigmoid(z)                                       # descalibrada respecto de q_true
    th = fit_platt(p_raw, y1)
    assert abs(th[0] - 1.7) < 0.15 and abs(th[1] - 0.4) < 0.15
    delta = rng.normal(0, 3000, n); r = np.where(y1, delta, rng.normal(0, 100, n))
    th2 = fit_rmse_calibration(p_raw, r, delta, with_size=False)
    sse_id = ((r - p_raw * delta) ** 2).sum(); sse_cal = ((r - apply_calibration(th2, p_raw, delta) * delta) ** 2).sum()
    assert sse_cal < sse_id
    th3 = fit_rmse_calibration(p_raw, r, delta, with_size=True)
    assert th3.size == 3 and ((r - apply_calibration(th3, p_raw, delta) * delta) ** 2).sum() <= sse_cal * 1.001
    assert np.allclose(logit(sigmoid(np.array([-3.0, 0.0, 2.0]))), [-3.0, 0.0, 2.0])


def _synthetic_inner(seed: int = 0, n: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    month = np.where(np.arange(n) < n // 2, 2, 8).astype(np.int8)
    s = rng.normal(2500, 2000, n); b = rng.normal(900, 150, n)
    cls = (rng.random(n) < 0.05).astype(np.int8)
    y = np.where(cls == 1, s, b + rng.normal(0, 120, n))
    p0 = np.clip(0.03 + 0.5 * (cls == 1) * rng.random(n), 0, 1)
    P0 = np.column_stack([1 - p0, p0, np.zeros(n)])
    in_cell = np.zeros(n, bool); in_cell[:20] = True
    P0[:20] = [0.2, 0.5, 0.3]
    return {"y": y, "s": s, "b0": b, "P0": P0, "p1n": p0, "month": month, "regime_class": cls, "in_cell": in_cell,
            "has_aobt3": np.ones(n, np.int8), "days_after_sched": np.zeros(n, np.int16), "mvt_minus_aobt3": b.copy(),
            "hour_utc": np.full(n, 10, np.int8), "airport": np.array(["EDDF"] * n)}


def test_cell_rows_and_clip():
    from v4b_stage2 import Candidate, cell_mixture, predict_full
    from v2_protocol import DAY_SEC
    d = _synthetic_inner()
    b, p = d["b0"], d["P0"][:, 1]
    cand = Candidate("S", 0).fit(d, np.arange(100, 200), b, p)
    rows = np.arange(0, 60)
    out = predict_full(d, cand, b, p, rows)
    cm = cell_mixture(d["P0"][:20], b[:20], d["s"][:20])
    assert np.allclose(out[:20], np.maximum(cm, 0))                       # celda → mezcla de 3 clases de V3
    assert np.allclose(cm, 0.2 * b[:20] + 0.5 * d["s"][:20] + 0.3 * (DAY_SEC + b[:20]))
    assert np.allclose(out[20:], np.maximum(b[20:60] + p[20:60] * (d["s"][20:60] - b[20:60]), 0))   # fuera: p2 = 0
    d["s"][30] = -1e6
    assert predict_full(d, cand, b, p, rows)[30] >= 0                      # recorte a ≥ 0


# ---------------------------------------------------------------------------
# 4. Sin fuga: la predicción interna de un mes no depende de su propio target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["S_T", "S_R2", "S_R3", "H"])
def test_crossfit_inner_no_leakage(variant):
    from v4b_stage2 import crossfit_inner
    d = _synthetic_inner(1)
    b, p = d["b0"], d["P0"][:, 1]
    base = crossfit_inner(d, variant, 0, b, p)
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    aug = d2["month"] == 8
    d2["y"][aug] = d2["y"][aug] + np.random.default_rng(9).normal(0, 5000, aug.sum())   # se altera solo agosto
    d2["regime_class"][aug] = 1 - d2["regime_class"][aug]
    pert = crossfit_inner(d2, variant, 0, b, p)
    assert np.allclose(base[aug], pert[aug])            # agosto se predice con el ajuste de febrero
    assert not np.allclose(base[~aug], pert[~aug]) or variant == "H"   # febrero sí cambia (control)


def test_selection_uses_only_inner_and_all_pairs():
    from v4b_stage2 import select
    rows = []
    for f in ("LOJO", "LOJuO"):
        for s in (42, 7):
            for v, sse in (("S", 10.0), ("S_R2", 8.0), ("H", 9.0)):
                rows.append({"fold": f, "seed": s, "config": "C1", "normal": "n0", "source": "p0", "variant": v, "inner_sse": sse})
                rows.append({"fold": f, "seed": s, "config": "C2", "normal": "n2", "source": "pn", "variant": v, "inner_sse": sse + (5 if v == "S_R2" else 0)})
    rows.append({"fold": "LOJO", "seed": 42, "config": "C2", "normal": "n2", "source": "pn", "variant": "D", "inner_sse": 0.0})  # incompleta
    sel = select(pl.from_dicts(rows))
    assert sel["C1"]["key"] == "C1__n0__p0__S_R2" and sel["C2"]["key"] == "C2__n2__pn__H"   # D descartada: no está en todas
    assert "outer" not in json.dumps(sel).lower()


# ---------------------------------------------------------------------------
# 5. Bootstrap ponderado y RMSE_w
# ---------------------------------------------------------------------------
def test_weighted_bootstrap_and_rmse_w():
    from v4b_analysis import W, rmse_w, weighted_block_bootstrap
    assert np.isclose(W["LOJO"] + W["LOJuO"], 1.0) and np.isclose(W["LOJO"], 152_719 / 344_841)
    assert np.isclose(rmse_w({"LOJO": 100.0, "LOJuO": 100.0}), 10.0)
    rng = np.random.default_rng(0)
    e = {f: {"A": rng.random(500) * 100, "B": rng.random(500) * 100} for f in W}
    e = {f: {**v, "A2": v["A"].copy()} for f, v in e.items()}
    blocks = {f: np.repeat(np.arange(50), 10).astype(str) for f in W}
    bt = weighted_block_bootstrap(e, blocks, 200, 1)
    assert np.allclose(bt["A"], bt["A2"])                 # mismas filas remuestreadas para todos los nombres
    assert bt["A"].shape == (200,)


# ---------------------------------------------------------------------------
# 6. Coherencia de las salidas y ausencia de candidata
# ---------------------------------------------------------------------------
def test_v4b_outputs_consistent():
    if not (V4B / "decision.json").exists():
        pytest.skip("aún no hay decisión V4b")
    dec = json.loads((V4B / "decision.json").read_text())
    d20 = dec["deltas"]["C2-C0"]
    assert dec["pass_to_costly_stage"] == (d20["delta_w"] <= dec["threshold_s"])
    assert dec["candidate_generated"] is False and dec["uploaded_to_bucket"] is False and dec["pushed"] is False
    assert dec["leaderboard_used"] is False
    assert not list((ROOT / "submissions").glob("candidate_v4*")) if (ROOT / "submissions").exists() else True
    sel = json.loads((V4B / "stage2_selection.json").read_text())
    assert dec["primary"]["C1"] == sel["C1"]["key"] and dec["primary"]["C2"] == sel["C2"]["key"]
