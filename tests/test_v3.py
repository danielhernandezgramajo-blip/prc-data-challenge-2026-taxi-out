"""
Pruebas V3 (etapa 2): feature de desplazamiento de fecha, clasificador con celdas (sin target en predicción,
actualización bayesiana exacta), alineación con V2, mezcla y reproducción de artefactos. Solo carpetas temporales.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, TARGET


# ---------------------------------------------------------------------------
# 1. days_after_sched: fecha(MVT) − fecha(SCHED), sin target ni BLOCK_TIME, idéntica en train y rank
# ---------------------------------------------------------------------------
def test_date_shift_feature(train_feats, rank_feats):
    from v3_protocol import DATE_SHIFT_COL, _DATE_RAW, add_date_shift
    assert "TAXITIME_SEC_mvt" not in _DATE_RAW and "BLOCK_TIME_UTC_mvt" not in _DATE_RAW
    tr = add_date_shift(train_feats.head(200_000), "train"); rk = add_date_shift(rank_feats.head(50_000), "rank")
    assert tr.height == 200_000 and rk.height == 50_000 and tr[DATE_SHIFT_COL].null_count() == 0 and rk[DATE_SHIFT_COL].null_count() == 0
    # coherencia con mvt_minus_sched: si MVT − SCHED < 0 el desplazamiento es ≤ 0; si ≥ 86 400 s es ≥ 1
    chk = tr.select(DATE_SHIFT_COL, "mvt_minus_sched")
    big = chk.filter(pl.col("mvt_minus_sched") >= 86_400); neg = chk.filter(pl.col("mvt_minus_sched") < 0)
    assert big.height == 0 or big[DATE_SHIFT_COL].min() >= 1
    assert neg.height == 0 or neg[DATE_SHIFT_COL].max() <= 0


# ---------------------------------------------------------------------------
# 2. Clasificador etapa 2: features permitidas, celdas exactas, predicción sin target, α = 0 ≡ sin celdas
# ---------------------------------------------------------------------------
def test_regime_classifier2_cells(small_train):
    from prepare_training import feature_columns
    from v2_protocol import regime_class_expr
    from v3_protocol import DATE_SHIFT_COL, RegimeClassifier2, add_date_shift, shifted_no_aobt3_expr
    tr = add_date_shift(small_train, "train")
    # forzamos una celda sintética: 40 filas LIRF sin AOBT_3 desplazadas, 30 de clase 1 (y = MVT − SCHED), 5 de clase 2
    n = 40
    forced = tr.head(n).with_columns(pl.lit("LIRF").alias("ADEP_mvt"), pl.lit(0).cast(tr["has_aobt3"].dtype).alias("has_aobt3"),
                                     pl.lit(1).cast(pl.Int16).alias(DATE_SHIFT_COL), pl.lit(None).cast(tr["mvt_minus_aobt3"].dtype).alias("mvt_minus_aobt3"))
    s = pl.col("mvt_minus_sched").cast(pl.Float64)
    y_new = pl.when(pl.int_range(0, n) < 30).then(s.round(0)).when(pl.int_range(0, n) < 35).then(pl.lit(86_400.0 + 900.0)).otherwise(s - 3000.0)
    forced = forced.with_columns(y_new.cast(tr[TARGET].dtype).alias(TARGET))
    tr2 = pl.concat([forced, tr.slice(n)], how="vertical")
    lab = tr2.select(regime_class_expr())["regime_class"].to_numpy()
    assert (lab[:30] == 1).all() and (lab[30:35] == 2).all() and (lab[35:40] == 0).all()
    m = RegimeClassifier2(FEATURE_NUM_COLS, FEATURE_CAT_COLS, iterations=20, alpha=20).fit(tr2)
    assert set(m.num + m.cat) == set(feature_columns()) | {DATE_SHIFT_COL}
    assert TARGET not in m.num + m.cat and "BLOCK_TIME_UTC_mvt" not in m.num + m.cat
    cell_train = tr2.filter(shifted_no_aobt3_expr() & (pl.col("ADEP_mvt") == "LIRF"))
    n_c, k1, k2 = m.cells["LIRF"]
    assert n_c == cell_train.height and k1 == int((cell_train.select(regime_class_expr())["regime_class"] == 1).sum()) and k2 == int((cell_train.select(regime_class_expr())["regime_class"] == 2).sum())
    val = tr2.head(500).drop(TARGET)                      # sin target: la predicción no debe necesitarlo
    p0 = m.predict_proba(val, alpha=0); p20 = m.predict_proba(val, alpha=20)
    for p in (p0, p20):
        assert p.shape == (500, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p >= 0).all()
    incell = val.select(shifted_no_aobt3_expr().alias("c"))["c"].to_numpy()
    assert incell[:n].all() and not incell[n:].any()
    # fuera de la celda α no cambia nada; dentro, la actualización bayesiana es exacta
    assert np.allclose(p0[~incell], p20[~incell])
    p1_raw, p2_tab, _ = m._raw(val)
    exp1 = (k1 + 20 * p1_raw[:n]) / (n_c + 20); exp2 = (k2 + 20 * p2_tab[:n]) / (n_c + 20)
    assert np.allclose(p20[:n, 1], exp1) and np.allclose(p20[:n, 2], exp2)
    assert p20[:n, 1].mean() > p0[:n, 1].mean() and p20[:n, 2].mean() > p0[:n, 2].mean()
    # modo jerárquico por banda (etapa 2c): fuera de la celda idéntico; dentro, segunda actualización con la banda de retraso
    from v3_protocol import BAND_SEC
    pb = m.predict_proba(val, alpha=20, mode="band")
    assert np.allclose(pb[~incell], p20[~incell])
    band = (val["mvt_minus_sched"].cast(pl.Float64).to_numpy()[:n] >= BAND_SEC).astype(int)
    for i in range(n):
        nb, kb1, kb2 = m.cells_band.get(("LIRF", band[i]), (0, 0, 0))
        assert abs(pb[i, 1] - (kb1 + 20 * exp1[i]) / (nb + 20)) < 1e-9 and abs(pb[i, 2] - (kb2 + 20 * exp2[i]) / (nb + 20)) < 1e-9
    assert sum(v[0] for k, v in m.cells_band.items() if k[0] == "LIRF") == n_c
    # etapa 2d: selección anidada del umbral SOLO con filas de entrenamiento; "nested" reproduce "flat" o la banda elegida
    from v3_protocol import BAND_GRID_H
    loo = m.loo
    assert set(loo["loo_logloss"]) == {"flat", *[f"band_{h}h" for h in BAND_GRID_H]} and loo["n_cell_rows"] >= n_c
    assert loo["selected_h"] is None or loo["selected_h"] in BAND_GRID_H
    pn = m.predict_proba(val, alpha=20, mode="nested")
    if loo["selected_h"] is None:
        assert np.allclose(pn, p20)
    else:
        assert loo["loo_logloss"][f"band_{loo['selected_h']}h"] < loo["loo_logloss"]["flat"]
        sec = loo["selected_h"] * 3600.0; table = m.cells_band_grid[loo["selected_h"]]
        band_sel = (val["mvt_minus_sched"].cast(pl.Float64).to_numpy()[:n] >= sec).astype(int)
        for i in range(n):
            nb, kb1, kb2 = table.get(("LIRF", band_sel[i]), (0, 0, 0))
            assert abs(pn[i, 1] - (kb1 + 20 * exp1[i]) / (nb + 20)) < 1e-9 and abs(pn[i, 2] - (kb2 + 20 * exp2[i]) / (nb + 20)) < 1e-9
    assert np.allclose(pn[~incell], p20[~incell])


# ---------------------------------------------------------------------------
# 3. Mezcla: identidades; alineación de ids con V2
# ---------------------------------------------------------------------------
def test_mixture_and_alignment():
    from v2_protocol import DAY_SEC, PRED_DIR as PRED_DIR_V2
    from v3_protocol import mixture
    normal = np.array([1000.0, 1000.0, 1000.0]); sched = np.array([5000.0, np.nan, 5000.0])
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(mixture(P, normal, sched), [1000.0, 0.0, DAY_SEC + 1000.0])   # NaN en MVT−SCHED → 0 en esa rama
    if not (PRED_DIR_V2 / "meta__May.npz").exists():
        pytest.skip("sin artefactos V2")
    from v3_protocol import load_features_v3, split_fold, _check_alignment
    feats = load_features_v3("train", surface=False)
    _, val = split_fold(feats, "May"); _check_alignment("May", val)


# ---------------------------------------------------------------------------
# 4. Artefactos V3: reproducen las métricas publicadas; diciembre solo con --holdout; regla de decisión
# ---------------------------------------------------------------------------
def test_v3_artifacts_consistent():
    from v2_protocol import DEV_FOLDS, HOLDOUT_FOLD
    from v3_protocol import OUT_DIR
    from v3_analysis import ELIGIBLE, candidates, load_fold
    from v2_metrics import rmse
    if not (OUT_DIR / "metrics_by_fold.csv").exists():
        pytest.skip("aún no se ha ejecutado v3_analysis")
    m = pl.read_csv(OUT_DIR / "metrics_by_fold.csv"); run = json.loads((OUT_DIR / "analysis_run.json").read_text())
    if not run["holdout_opened"]:
        assert m.filter(pl.col("fold") == HOLDOUT_FOLD).height == 0
    for f in DEV_FOLDS:
        d = load_fold(f); c = candidates(d)
        for cand in ("V1P", "S_mix", "S2_mix", "S2A_mix", "S3A_mix", "S4A_mix"):
            if cand in c:
                pub = m.filter((pl.col("fold") == f) & (pl.col("candidate") == cand))["rmse"][0]
                assert abs(rmse(d["y"], c[cand]) - pub) < 1e-6
    dec = json.loads((OUT_DIR / "decision_dev.json").read_text())
    sel = pl.read_csv(OUT_DIR / "selection_criteria.csv")
    for c in dec["eligible"]:
        ok = all(sel.filter((pl.col("candidate") == c) & (pl.col("vs") == b))["all_required_criteria"][0] for b in ("S_mix", "V1P"))
        assert ok == (c in [p[1] for p in dec["passing"]])
    if dec["finalist"] is not None:
        assert dec["finalist"] in ELIGIBLE and dec["finalist"] == min(dec["passing"])[1]
    # si hay una elección final distinta de la finalista congelada, debe estar documentada y cumplir los criterios en desarrollo
    if (OUT_DIR / "final_choice.json").exists():
        fc = json.loads((OUT_DIR / "final_choice.json").read_text())
        assert fc["reason"] and all(sel.filter((pl.col("candidate") == fc["candidate"]) & (pl.col("vs") == b))["all_required_criteria"][0] for b in ("S_mix", "V1P"))
        if fc["candidate"] != json.loads((OUT_DIR / "finalist_frozen.json").read_text())["finalist"]:
            assert fc["post_hoc"] is True
