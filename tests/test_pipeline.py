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
#    Se verifica: (a) la unión es por MVT_ID_mvt y no por posición (predicciones barajadas → cada fila recibe la
#    predicción de su id); (b) el orden y el esquema del resultado son los del template (Int32); (c) antes del join,
#    assemble_submission() rechaza con ValueError: IDs faltantes, un ID inexistente en el template (adicional),
#    IDs duplicados, IDs nulos y columnas incorrectas; (d) validate_submission() rechaza un archivo truncado;
#    (e) ida y vuelta por Parquet; (f) si existe candidate_v1.parquet, pasa todas las validaciones y no fue subida.
# ---------------------------------------------------------------------------
def test_submission_alignment(rank_feats, tmp_path):
    from io_utils import read_submitting
    from make_submission import assemble_submission, check_prediction_ids, validate_submission
    template = read_submitting()
    rng = np.random.default_rng(0)
    pred = pl.DataFrame({ID_COL: rank_feats[ID_COL], "pred": rng.uniform(300, 2000, rank_feats.height)})
    shuffled = pred.sample(fraction=1.0, shuffle=True, seed=1)
    info = check_prediction_ids(shuffled, template)
    assert info == {"n": template.height, "n_null_ids": 0, "n_duplicate_ids": 0, "n_missing": 0, "n_extra": 0}
    sub = assemble_submission(shuffled, template)
    checks = validate_submission(sub, template)
    assert checks["ok"] and checks["order_identical"] and checks["schema_ok"]
    assert sub.schema[TARGET] == template.schema[TARGET] == pl.Int32
    # (a) la unión es por id: la predicción de cada fila coincide con la de su id, no con la posición
    j = sub.join(pred, on=ID_COL, how="left")
    assert bool((j[TARGET] == j["pred"].round(0).cast(pl.Int32)).all())
    # (c1) IDs faltantes → ValueError con la cantidad exacta
    with pytest.raises(ValueError, match=r"343841 faltantes, 0 adicionales"):
        assemble_submission(pred.head(1000), template)
    # (c2) un ID inexistente en el template (adicional) → ValueError aunque el resto esté completo
    fake_id = float(pred[ID_COL].max() + 1_000_000)
    with_extra = pl.concat([pred, pl.DataFrame({ID_COL: [fake_id], "pred": [900.0]}, schema=pred.schema)])
    with pytest.raises(ValueError, match=r"0 faltantes, 1 adicionales"):
        assemble_submission(with_extra, template)
    # (c3) IDs duplicados → ValueError
    with_dup = pl.concat([pred, pred.head(3)])
    with pytest.raises(ValueError, match=r"3 IDs duplicados"):
        assemble_submission(with_dup, template)
    # (c4) IDs nulos → ValueError
    with_null = pred.with_columns(pl.when(pl.arange(0, pl.len()) < 2).then(None).otherwise(pl.col(ID_COL)).alias(ID_COL))
    with pytest.raises(ValueError, match=r"2 IDs nulos"):
        assemble_submission(with_null, template)
    # (c5) columnas incorrectas → ValueError
    with pytest.raises(ValueError, match=r"exactamente las columnas"):
        assemble_submission(pred.rename({"pred": "prediction"}), template)
    with pytest.raises(ValueError, match=r"exactamente las columnas"):
        assemble_submission(pred.with_columns(pl.lit(1).alias("extra")), template)
    # (d) validate_submission rechaza un archivo truncado
    with pytest.raises(AssertionError):
        validate_submission(sub.head(1000), template)
    # (e) ida y vuelta por Parquet
    sub.write_parquet(tmp_path / "s.parquet")
    assert validate_submission(pl.read_parquet(tmp_path / "s.parquet"), template)["ok"]
    # (g) serialización determinista: un solo chunk por columna y el mismo Parquet (byte a byte) aunque las
    #     predicciones lleguen con distinta fragmentación física; orden, esquema y valores intactos
    assert sub.n_chunks("all") == [1] * sub.width
    fragmented = pl.concat([shuffled.slice(i, 50_000) for i in range(0, shuffled.height, 50_000)], rechunk=False)
    assert fragmented.n_chunks() > 1
    sub2 = assemble_submission(fragmented, template)
    assert sub2.n_chunks("all") == [1] * sub2.width
    sub2.write_parquet(tmp_path / "s2.parquet")
    assert hashlib.sha256((tmp_path / "s.parquet").read_bytes()).hexdigest() == hashlib.sha256((tmp_path / "s2.parquet").read_bytes()).hexdigest()
    assert sub2.equals(sub) and sub2.schema == template.schema and bool((sub2[ID_COL] == template[ID_COL]).all())
    # (f) si existe la candidata real, también debe pasar y no haber sido subida
    cand = ROOT / "submissions" / "candidate_v1.parquet"
    if cand.exists():
        c = validate_submission(pl.read_parquet(cand), template)
        assert c["ok"] and c["n_negative"] == 0
        meta = json.loads(cand.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["uploaded_to_bucket"] is False


# ---------------------------------------------------------------------------
# 9. Auditoría del conjunto exacto de unidades (fold, strategy, model) del manifiesto
# ---------------------------------------------------------------------------
def test_audit_exact_units():
    import backtest
    manifest_path = TABLES_DIR / "backtest_run_args.json"
    if not manifest_path.exists() or not (TABLES_DIR / "backtest_metrics.csv").exists():
        pytest.skip("aún no hay una corrida de backtest")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pl.read_csv(TABLES_DIR / "backtest_metrics.csv")
    # el manifiesto actual espera exactamente 84 unidades y las 84 observadas coinciden
    assert len(backtest.expected_units_from_manifest(man)) == 84
    a = backtest.audit_manifest(require_model=True, metrics=metrics, manifest=man)
    assert a["expected_units"] == 84 and a["observed_units"] == 84
    assert a["missing_units"] == [] and a["unexpected_units"] == [] and a["consistent"]
    # una combinación completa eliminada en memoria → missing_units e inconsistente (aunque los marginales coincidan)
    drop = (pl.col("fold") == "F1_jul") & (pl.col("strategy") == "drop_gt3h") & (pl.col("model") == "catboost_direct")
    a2 = backtest.audit_manifest(require_model=True, metrics=metrics.filter(~drop), manifest=man)
    assert a2["missing_units"] == ["F1_jul/drop_gt3h/catboost_direct"] and a2["unexpected_units"] == []
    assert a2["observed_units"] == 83 and not a2["consistent"]
    assert not any("!= observad" in i for i in a2["issues"]), "los marginales siguen coincidiendo; solo debe fallar el conjunto exacto"
    # una combinación no esperada (F1_jul + drop_gt3h + global_median duplicada como estrategia inventada) → unexpected_units
    extra = metrics.head(1).with_columns(pl.lit("F1_jul").alias("fold"), pl.lit("estrategia_inventada").alias("strategy"), pl.lit("global_median").alias("model"))
    a3 = backtest.audit_manifest(require_model=True, metrics=pl.concat([metrics, extra], how="vertical_relaxed"), manifest=man)
    assert a3["unexpected_units"] == ["F1_jul/estrategia_inventada/global_median"] and not a3["consistent"]
    # cb_strategy_fold añade todas las estrategias solo a ese fold, sin duplicados
    man2 = {**man, "cb_strategies": ["original"], "cb_strategy_fold": "F2_nov"}
    exp2 = backtest.expected_units_from_manifest(man2)
    assert len(exp2) == 60 + 3 * 1 * 2 + 3 * 2  # baselines + CatBoost original en 3 folds + 3 estrategias extra en F2 × 2 modelos
