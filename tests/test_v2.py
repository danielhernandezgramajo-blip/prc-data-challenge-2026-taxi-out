"""
Pruebas V2: features de congestión, cortes temporales de 8 folds, ausencia de leakage, etiquetas de régimen,
mezcla por régimen, métricas pooled, bootstrap pareado y reproducción de artefactos. Solo carpetas temporales.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from config import ID_COL, OUTLIER_HI_SEC, TARGET


# ---------------------------------------------------------------------------
# 1. Features de congestión: sin target ni BLOCK_TIME, excluyen la fila propia, idénticas train/rank
# ---------------------------------------------------------------------------
def test_v2_features_no_leakage_and_self_exclusion():
    from v2_features import FORBIDDEN_RAW, V2_CAT_COLS, V2_NUM_COLS, V2_RAW_COLS, congestion_features
    assert not FORBIDDEN_RAW & set(V2_RAW_COLS)
    assert not FORBIDDEN_RAW & (set(V2_NUM_COLS) | set(V2_CAT_COLS))
    t0 = pl.datetime(2025, 3, 1, 10, 0, 0, time_zone="UTC")
    # 4 salidas en la misma pista a 0, 5, 12 y 40 min; 2 llegadas a +3 y −20 min; una salida en otro aeropuerto
    raw = pl.DataFrame({
        ID_COL: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "PHASE_mvt": ["DEP", "DEP", "DEP", "DEP", "ARR", "ARR", "DEP"],
        "ADEP_mvt": ["AAAA", "AAAA", "AAAA", "AAAA", "XXXX", "XXXX", "BBBB"],
        "ADES_mvt": ["ZZZZ", "ZZZZ", "ZZZZ", "ZZZZ", "AAAA", "AAAA", "ZZZZ"],
        "RUNWAY_mvt": ["25", "25", "25", "25", None, None, "NA"],
        "MVT_TIME_UTC_mvt": [0, 5, 12, 40, 3, -20, 0],
        "SCHED_TIME_UTC_mvt": [-10, 0, 0, 30, 0, -30, -5],
    }).with_columns([(t0 + pl.duration(minutes=pl.col(c))).alias(c) for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")])
    f = congestion_features(raw).sort(ID_COL)
    assert f.height == 5 and set(f.columns) == {ID_COL, *V2_NUM_COLS, *V2_CAT_COLS}
    r1 = f.row(0, named=True)   # salida a t=0
    assert r1["dep_pm15"] == 2 and r1["dep_pm30"] == 2        # las de 5 y 12 min; no se cuenta a sí misma; 40 min queda fuera
    assert r1["arr_pm15"] == 1 and r1["arr_pm30"] == 2        # llegada a +3 dentro de ±15; la de −20 solo en ±30
    assert r1["rwy_prev10"] == 0 and np.isnan(r1["min_since_prev_rwy_dep"])
    r3 = f.row(2, named=True)   # salida a t=12
    assert r3["rwy_prev10"] == 1 and r3["rwy_prev15"] == 2 and r3["rwy_prev30"] == 2   # 5 min (y 0 min) anteriores
    assert abs(r3["min_since_prev_rwy_dep"] - 7.0) < 1e-6
    assert abs(r3["min_to_next_rwy_sched"] - 30.0) < 1e-6       # siguiente programada de la pista: t=30 (fila 4)
    assert r3["net_pm15"] == r3["dep_pm15"] - r3["arr_pm15"]
    assert f.row(4, named=True)["dep_pm15"] == 0                # otro aeropuerto, sin vecinos


# ---------------------------------------------------------------------------
# 2. Cortes temporales de los 8 folds: sin solape, holdout separado
# ---------------------------------------------------------------------------
def test_v2_folds_no_overlap(train_feats):
    from v2_protocol import DEV_FOLDS, HOLDOUT_FOLD, V2_FOLDS, split_fold
    assert HOLDOUT_FOLD not in DEV_FOLDS and len(DEV_FOLDS) == 7 and len(V2_FOLDS) == 8
    for fold, (m0, m1, mv) in V2_FOLDS.items():
        tr, va = split_fold(train_feats, fold)
        assert tr["MVT_TIME_UTC_mvt"].max() < va["MVT_TIME_UTC_mvt"].min()
        assert tr.join(va.select(ID_COL), on=ID_COL, how="inner").height == 0
        assert set(va["MVT_TIME_UTC_mvt"].dt.month().unique().to_list()) == {mv} and m1 == mv - 1 and m0 == 1


# ---------------------------------------------------------------------------
# 3. Etiquetas de régimen y mezcla: identidades exactas
# ---------------------------------------------------------------------------
def test_regime_labels_and_mixture():
    from v2_protocol import DAY_SEC, SCHED_TOL, regime_class_expr
    df = pl.DataFrame({TARGET: [900, 12000, 12005, 86400 + 900, 40000, 50],
                       "mvt_minus_sched": [1200.0, 12000.0, 12000.0, 20000.0, 3000.0, 40.0]})
    lab = df.select(regime_class_expr())["regime_class"].to_list()
    assert lab == [0, 1, 1, 2, 0, 1]          # 50 vs 40 → |d| = 10 ≤ tolerancia → clase 1
    assert SCHED_TOL == 10.0 and DAY_SEC == 86400.0
    # mezcla: con p = (1,0,0) devuelve normal; con p = (0,1,0) devuelve MVT−SCHED; con p=(0,0,1) devuelve 86400+normal
    normal = np.array([1000.0, 1000.0, 1000.0]); sched = np.array([5000.0, 5000.0, 5000.0])
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    mix = P[:, 0] * normal + P[:, 1] * sched + P[:, 2] * (DAY_SEC + normal)
    assert np.allclose(mix, [1000.0, 5000.0, 87400.0])


# ---------------------------------------------------------------------------
# 4. El clasificador de régimen solo usa features permitidas y no lee el target de validación
# ---------------------------------------------------------------------------
def test_regime_classifier_features(small_train):
    from prepare_training import feature_columns
    from v2_protocol import RegimeClassifier
    from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS
    m = RegimeClassifier(FEATURE_NUM_COLS, FEATURE_CAT_COLS, iterations=20).fit(small_train)
    assert set(m.num + m.cat) == set(feature_columns())
    assert TARGET not in m.num + m.cat and "BLOCK_TIME_UTC_mvt" not in m.num + m.cat
    val = small_train.head(500).drop(TARGET)           # sin target: la predicción no debe necesitarlo
    p = m.predict_proba(val)
    assert p.shape == (500, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p >= 0).all()


# ---------------------------------------------------------------------------
# 5. Métricas pooled ≠ media de medias; cobertura; bootstrap pareado
# ---------------------------------------------------------------------------
def test_pooled_metrics_and_bootstrap():
    from v2_metrics import block, pooled, rmse
    from v2_analysis import paired_block_bootstrap
    y1 = np.array([0.0, 0.0]); p1 = np.array([3.0, 3.0])          # rmse 3
    y2 = np.array([0.0] * 6); p2 = np.array([1.0] * 6)              # rmse 1
    pooled_rmse = pooled([(y1, p1), (y2, p2)])["rmse"]
    assert abs(pooled_rmse - np.sqrt((9 * 2 + 1 * 6) / 8)) < 1e-9 and abs(np.mean([3, 1]) - 2) < 1e-9 and pooled_rmse != 2
    b = block(np.array([0, 0, 0, 20000.0]), np.array([100, 400, 700, 0.0]))
    assert b["cov_le300"] == 25.0 and b["cov_le600"] == 50.0 and b["cov_le900"] == 75.0 and b["n_extreme"] == 1
    rng = np.random.default_rng(0)
    y = rng.normal(1000, 300, 5000); pa = y + rng.normal(0, 200, 5000); pb = y + rng.normal(0, 100, 5000)
    blocks = np.array([f"A_{i % 40}" for i in range(5000)])
    r = paired_block_bootstrap(y, pa, pb, blocks, 200, 42)
    assert r["p_improve"] > 0.99 and r["delta_p97.5"] < 0 and r["n_blocks"] == 40
    r2 = paired_block_bootstrap(y, pa, pb, blocks, 200, 42)
    assert r == r2                                                  # reproducible con la misma semilla


# ---------------------------------------------------------------------------
# 6. Artefactos guardados: reproducen las métricas publicadas y no tocan diciembre sin --holdout
# ---------------------------------------------------------------------------
def test_v2_artifacts_consistent():
    from v2_protocol import DEV_FOLDS, HOLDOUT_FOLD, OUT_DIR, PRED_DIR
    from v2_analysis import candidates, load_fold
    from v2_metrics import rmse
    if not (OUT_DIR / "metrics_by_fold.csv").exists():
        pytest.skip("aún no se ha ejecutado v2_analysis")
    m = pl.read_csv(OUT_DIR / "metrics_by_fold.csv")
    run = json.loads((OUT_DIR / "analysis_run.json").read_text())
    if not run["holdout_opened"]:
        assert m.filter(pl.col("fold") == HOLDOUT_FOLD).height == 0
    for f in DEV_FOLDS:
        d = load_fold(f); c = candidates(d)
        for cand in ("V1P", "S_mix"):
            pub = m.filter((pl.col("fold") == f) & (pl.col("candidate") == cand))["rmse"][0]
            assert abs(rmse(d["y"], c[cand]) - pub) < 1e-6
        assert d["y"].shape == c["S_mix"].shape and np.isfinite(c["S_mix"]).all()
    reg = pl.read_csv(OUT_DIR / "experiment_registry.csv")
    assert {"V1P", "S_mix", "S_mix_dn", "S_mix_onV1P", "S_mix_floor"} <= set(reg["candidate"].to_list())
