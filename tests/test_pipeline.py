"""
Pruebas automatizadas del pipeline (pytest). Ejecutar desde la raíz del proyecto:
    python -m pytest tests -q
Las pruebas leen los Parquet originales (solo lectura) y escriben únicamente en carpetas temporales.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config import FEATURE_CAT_COLS, FOLDS, FORBIDDEN_FEATURES, ID_COL, TABLES_DIR, TARGET
from modeling_utils import CategoryCodec, apply_outlier_strategy

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Sin BLOCK_TIME ni target entre los predictores
# ---------------------------------------------------------------------------
def test_no_forbidden_predictors(train_feats):
    from prepare_training import RAW_COLS, assert_no_forbidden, feature_columns
    feats = feature_columns()
    assert not set(feats) & set(FORBIDDEN_FEATURES)
    assert "BLOCK_TIME_UTC_mvt" not in feats and TARGET not in feats and ID_COL not in feats
    assert "BLOCK_TIME_UTC_mvt" not in RAW_COLS, "BLOCK_TIME no debe ni siquiera leerse"
    assert "BLOCK_TIME_UTC_mvt" not in train_feats.columns
    assert_no_forbidden(train_feats)
    # el modelo CatBoost solo ve `features`
    from train_baseline import CatBoostDirect
    assert CatBoostDirect().features == feats


# ---------------------------------------------------------------------------
# 2. Esquema idéntico train / ranking
# ---------------------------------------------------------------------------
def test_schema_train_rank(train_feats, rank_feats):
    from prepare_training import check_same_schema
    check_same_schema(train_feats, rank_feats)
    assert train_feats.columns == rank_feats.columns
    assert [str(t) for t in train_feats.dtypes] == [str(t) for t in rank_feats.dtypes]
    assert rank_feats[TARGET].null_count() == rank_feats.height
    assert rank_feats.height == 344_841


# ---------------------------------------------------------------------------
# 3. Codec save / load
# ---------------------------------------------------------------------------
def test_codec_save_load(small_train, tmp_path):
    codec = CategoryCodec(FEATURE_CAT_COLS).fit(small_train)
    codec.save(tmp_path / "codec.json")
    loaded = CategoryCodec.load(tmp_path / "codec.json")
    assert loaded == codec
    a = codec.transform(small_train.select(FEATURE_CAT_COLS))
    b = loaded.transform(small_train.select(FEATURE_CAT_COLS))
    assert a.equals(b)
    # categoría desconocida → -1
    unk = small_train.head(1).with_columns(pl.lit("__NUEVA__").alias("STAND_mvt"))
    assert loaded.transform(unk)["STAND_mvt"][0] == -1


# ---------------------------------------------------------------------------
# 4. Bundle save / load con predicciones idénticas (directo y residual)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["cb_direct", "cb_residual"])
def test_bundle_roundtrip(small_train, rank_feats, tmp_path, model_key):
    from train_baseline import MODEL_REGISTRY, ModelBundle
    m = MODEL_REGISTRY[model_key](fixed_iterations=30)
    m.fit(small_train)
    check = rank_feats.head(5_000)
    before = m.predict(check)
    bdir = tmp_path / model_key
    manifest = ModelBundle(m, "drop_nonpositive").save(bdir, check)
    for f in ["model.cbm", "codec.json", "manifest.json", "check_predictions.npz"]:
        assert (bdir / f).exists()
    for k in ["features", "cat_cols", "strategy", "iterations", "seed", "versions", "trained_at_utc", "input_files"]:
        assert k in manifest
    assert manifest["iterations"] == 30 and manifest["seed"] == 42
    assert len(manifest["input_files"]) == 14 and all(len(f["sha256"]) == 64 for f in manifest["input_files"])
    loaded, _ = ModelBundle.load(bdir)
    after = loaded.predict(check)
    assert np.array_equal(before, after), f"max diff {np.max(np.abs(before - after))}"
    ver = ModelBundle.verify(bdir, rank_feats)
    assert ver["ok"] and ver["sha256_match"]


# ---------------------------------------------------------------------------
# 5. Cortes temporales sin solape
# ---------------------------------------------------------------------------
def test_split_fold_no_overlap(train_feats):
    from backtest import split_fold
    for fold, (m0, m1, mv) in FOLDS.items():
        tr, va = split_fold(train_feats, fold)
        assert tr["MVT_TIME_UTC_mvt"].max() < va["MVT_TIME_UTC_mvt"].min()
        assert tr.join(va.select(ID_COL), on=ID_COL, how="inner").height == 0
        assert set(va["MVT_TIME_UTC_mvt"].dt.month().unique().to_list()) == {mv}
        assert tr["MVT_TIME_UTC_mvt"].dt.month().max() == m1 and tr["MVT_TIME_UTC_mvt"].dt.month().min() == m0
    # la estrategia drop_nonpositive elimina exactamente las filas con target <= 0
    tr, _ = split_fold(train_feats, "F1_jul")
    out, info = apply_outlier_strategy(tr, "drop_nonpositive")
    assert out.height == tr.filter(pl.col(TARGET) > 0).height == info["n_train_after"]
    assert (out[TARGET] > 0).all()


# ---------------------------------------------------------------------------
# 6. --figures-only no modifica el manifiesto
# ---------------------------------------------------------------------------
def test_figures_only_preserves_manifest(monkeypatch, tmp_path):
    import backtest
    manifest = TABLES_DIR / "backtest_run_args.json"
    if not manifest.exists() or not (TABLES_DIR / "backtest_metrics.csv").exists():
        pytest.skip("aún no hay una corrida de backtest")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    # las figuras se escriben en una carpeta temporal para no tocar outputs/figures
    monkeypatch.setattr(backtest, "FIGURES_DIR", tmp_path / "figs")
    monkeypatch.setattr(backtest, "TABLES_DIR", tmp_path / "tables")
    shutil.copytree(TABLES_DIR, tmp_path / "tables")
    class A: figures_only = True
    backtest.finalize(A(), write_manifest=False)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    assert (tmp_path / "figs" / "model_rmse_comparison.png").exists()
    # y con write_manifest=True SÍ lo escribiría (en la copia temporal)
    backtest.finalize(A(), write_manifest=True)
    assert (tmp_path / "tables" / "backtest_run_args.json").exists()


# ---------------------------------------------------------------------------
# 7. Consistencia de todos los artefactos de --resume
# ---------------------------------------------------------------------------
def test_resume_artifacts_consistent():
    import backtest
    if not (TABLES_DIR / "backtest_metrics.csv").exists():
        pytest.skip("aún no hay una corrida de backtest")
    audit = backtest.audit_artifacts(require_model=True)
    assert audit["incomplete"] == {}, audit["incomplete"]
    assert all(v == 0 for v in audit["duplicates"].values()), audit["duplicates"]
    fi = pl.read_csv(TABLES_DIR / "feature_importance.csv")
    n_cb = fi.select("fold", "strategy", "model").unique().height
    assert fi.height == n_cb * backtest.N_FEATURES
    if audit["n_units"] == 84:  # corrida completa: 2 modelos × 3 folds × 4 estrategias × 29 features
        assert fi.height == 696
    fb = pl.read_csv(TABLES_DIR / "fallback_coverage.csv")
    expected_fb = {"hierarchical_median": 5, "operational_aobt3": 1, "operational_aobt3_biascorr": 1, "catboost_residual": 1}
    m = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    for r in m.select("fold", "strategy", "model").unique().iter_rows(named=True):
        n = fb.filter((pl.col("fold") == r["fold"]) & (pl.col("strategy") == r["strategy"]) & (pl.col("model") == r["model"])).height
        assert n == expected_fb.get(r["model"], 0), (r, n)
    assert fb.select("fold", "strategy", "model", "fallback").unique().height == fb.height


# ---------------------------------------------------------------------------
# 8. Alineación exacta de una submission con submitting.parquet
# ---------------------------------------------------------------------------
def test_submission_alignment(rank_feats, tmp_path):
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    template = read_submitting()
    rng = np.random.default_rng(0)
    pred = pl.DataFrame({ID_COL: rank_feats[ID_COL], "pred": rng.uniform(300, 2000, rank_feats.height)})
    shuffled = pred.sample(fraction=1.0, shuffle=True, seed=1)
    sub = assemble_submission(shuffled, template)
    checks = validate_submission(sub, template)
    assert checks["ok"] and checks["order_identical"] and checks["schema_ok"]
    assert sub.schema[TARGET] == template.schema[TARGET] == pl.Int32
    # la unión es por id: la predicción de cada fila coincide con la de su id, no con la posición
    j = sub.join(pred, on=ID_COL, how="left")
    assert bool((j[TARGET] == j["pred"].round(0).cast(pl.Int32)).all())
    # se detectan ids faltantes / de más
    with pytest.raises(ValueError):
        assemble_submission(pred.head(1000), template)
    with pytest.raises(AssertionError):
        validate_submission(sub.head(1000), template)
    # ida y vuelta por Parquet
    sub.write_parquet(tmp_path / "s.parquet")
    assert validate_submission(pl.read_parquet(tmp_path / "s.parquet"), template)["ok"]
    # si existe la candidata real, también debe pasar
    cand = ROOT / "submissions" / "candidate_v1.parquet"
    if cand.exists():
        c = validate_submission(pl.read_parquet(cand), template)
        assert c["ok"] and c["n_negative"] == 0
        meta = json.loads(cand.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["uploaded_to_bucket"] is False
