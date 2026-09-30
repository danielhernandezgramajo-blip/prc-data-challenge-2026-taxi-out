"""
V3 — entrenamiento final de la finalista de la etapa 2 con todo 2025 y generación de submissions/candidate_v3.parquet.

    python src/v3_final.py train      # bundles en models/experiments/v3/final/ (no toca models/final ni models/experiments/v2)
    python src/v3_final.py submit     # candidate_v3.parquet desde los bundles, sin reentrenar; validación idéntica a V1/V2

La finalista se lee de outputs/experiments/v3/finalist_frozen.json (regla prefijada de v3_analysis). Componentes:
    proba   : "proba2"  → RegimeClassifier2 (α = 20) entrenado aquí con todo 2025 (drop_nonpositive)
              "proba"   → clasificador final de V2 (models/experiments/v2/final/regime_classifier), sin reentrenar
    normal  : media simple de los componentes que defina la finalista, entre
              direct_normal (bundle final V2), residual_V1 (models/final/catboost_residual, congelado),
              direct_normal_s y residual_normal_s (entrenados aquí con features V1 + congestión, filas clase 0 y y ≤ 3 h)
    pred = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal); negativos → 0 (como V1/V2).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, MODELS_DIR, OUTLIER_HI_SEC, PROJECT_ROOT, SUBMISSIONS_DIR, TARGET  # noqa: E402
from io_utils import read_submitting  # noqa: E402
from make_submission import assemble_submission, validate_submission  # noqa: E402
from modeling_utils import CategoryCodec, apply_outlier_strategy  # noqa: E402
from train_baseline import CatBoostResidual, HierarchicalMedian, ModelBundle, env_versions, input_inventory  # noqa: E402
from v2_final import load_classifier as load_classifier_v2  # noqa: E402
from v2_final import verify_classifier as verify_classifier_v2  # noqa: E402
from v2_protocol import STRATEGY, make_model, regime_class_expr  # noqa: E402
from v3_protocol import ALPHA_PRIMARY, BAND_SEC, DATE_SHIFT_COL, OUT_DIR, RegimeClassifier2, load_features_v3, mixture, variants  # noqa: E402

FINAL_DIR = MODELS_DIR / "experiments" / "v3" / "final"
V2_FINAL = MODELS_DIR / "experiments" / "v2" / "final"
V1_RESIDUAL = MODELS_DIR / "final" / "catboost_residual"
COMPONENTS = {  # candidata → (proba, componentes de la rama normal); proba2 = celdas planas, proba2_band = celdas jerárquicas por banda
    "S2_mix": ("proba2", ["direct_normal", "residual_v1"]),
    "A_mix": ("proba", ["direct_normal_s", "residual_normal_s"]),
    "A_mix_4": ("proba", ["direct_normal", "residual_v1", "direct_normal_s", "residual_normal_s"]),
    "S2A_mix": ("proba2", ["direct_normal_s", "residual_normal_s"]),
    "S2A_mix_4": ("proba2", ["direct_normal", "residual_v1", "direct_normal_s", "residual_normal_s"]),
    "S3_mix": ("proba2_band", ["direct_normal", "residual_v1"]),
    "S3A_mix": ("proba2_band", ["direct_normal_s", "residual_normal_s"]),
    "S3A_mix_4": ("proba2_band", ["direct_normal", "residual_v1", "direct_normal_s", "residual_normal_s"]),
    "S4_mix": ("proba2_nested", ["direct_normal", "residual_v1"]),
    "S4A_mix": ("proba2_nested", ["direct_normal_s", "residual_normal_s"]),
    "S4A_mix_4": ("proba2_nested", ["direct_normal", "residual_v1", "direct_normal_s", "residual_normal_s"]),
}
PROBA_MODE = {"proba2": "flat", "proba2_band": "band", "proba2_nested": "nested"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalist() -> tuple[str, str, list[str]]:
    """Candidata que va a candidate_v3: final_choice.json (elección documentada, puede ser post-hoc) si existe;
    si no, la finalista congelada por la regla prefijada (finalist_frozen.json)."""
    fc = OUT_DIR / "final_choice.json"
    if fc.exists():
        name = json.loads(fc.read_text(encoding="utf-8"))["candidate"]
    else:
        name = json.loads((OUT_DIR / "finalist_frozen.json").read_text(encoding="utf-8"))["finalist"]
    assert name in COMPONENTS, f"candidata {name!r} sin definición de componentes"
    return name, *COMPONENTS[name]


# ---------------------------------------------------------------------------
# Bundles: clasificador etapa 2 y regresores con features arbitrarias
# ---------------------------------------------------------------------------
def save_classifier2(m: RegimeClassifier2, bdir: Path, check_df: pl.DataFrame) -> dict:
    bdir.mkdir(parents=True, exist_ok=True)
    m.clf.save_model(str(bdir / "model.cbm")); m.codec.save(bdir / "codec.json")
    (bdir / "rate2.json").write_text(json.dumps({"global": m.rate2_global, "pseudo": 200,
        "table": [{"ADEP_mvt": k[0], "has_aobt3": k[1], "shifted": k[2], "rate": v} for k, v in m.rate2.items()]}, indent=1), encoding="utf-8")
    (bdir / "cells.json").write_text(json.dumps({"alpha": m.alpha, "cells": {k: {"n": v[0], "k1": v[1], "k2": v[2]} for k, v in m.cells.items()},
        "cells_band_grid": {str(h): [{"ADEP_mvt": k[0], "band": k[1], "n": v[0], "k1": v[1], "k2": v[2]} for k, v in t.items()] for h, t in m.cells_band_grid.items()},
        "band_sec": BAND_SEC, "band_sec_selected": m.band_sec_selected, "nested_selection": m.loo}, indent=1), encoding="utf-8")
    p = np.column_stack([m.predict_proba(check_df, mode="flat"), m.predict_proba(check_df, mode="band"), m.predict_proba(check_df, mode="nested")])
    np.savez_compressed(bdir / "check_predictions.npz", ids=check_df[ID_COL].to_numpy(), proba=p)
    man = {"model_name": "regime_classifier2", "features": m.num + m.cat, "num_cols": m.num, "cat_cols": m.cat, "iterations": m.iterations,
           "alpha": m.alpha, "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": env_versions(),
           "input_files": input_inventory(), "info": m.info, "check": {"n_rows": check_df.height, "proba_sha256": hashlib.sha256(p.tobytes()).hexdigest()}}
    (bdir / "manifest.json").write_text(json.dumps(man, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return man


def load_classifier2(bdir: Path) -> RegimeClassifier2:
    from catboost import CatBoostClassifier
    man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    num = [c for c in man["num_cols"] if c != DATE_SHIFT_COL]
    m = RegimeClassifier2(num, man["cat_cols"], iterations=man["iterations"], alpha=man["alpha"])
    assert m.num + m.cat == man["features"]
    m.codec = CategoryCodec.load(bdir / "codec.json")
    m.clf = CatBoostClassifier(); m.clf.load_model(str(bdir / "model.cbm"))
    r = json.loads((bdir / "rate2.json").read_text(encoding="utf-8"))
    m.rate2 = {(t["ADEP_mvt"], int(t["has_aobt3"]), int(t["shifted"])): t["rate"] for t in r["table"]}; m.rate2_global = r["global"]
    c = json.loads((bdir / "cells.json").read_text(encoding="utf-8"))
    m.cells = {k: (v["n"], v["k1"], v["k2"]) for k, v in c["cells"].items()}
    m.cells_band_grid = {int(h): {(t["ADEP_mvt"], int(t["band"])): (t["n"], t["k1"], t["k2"]) for t in rows} for h, rows in c["cells_band_grid"].items()}
    m.cells_band = m.cells_band_grid[int(BAND_SEC // 3600)]
    m.band_sec_selected = c["band_sec_selected"]; m.loo = c["nested_selection"]
    assert c["band_sec"] == BAND_SEC
    return m


def verify_classifier2(bdir: Path, df: pl.DataFrame) -> dict:
    m = load_classifier2(bdir)
    chk = np.load(bdir / "check_predictions.npz")
    order = pl.DataFrame({ID_COL: chk["ids"], "_o": np.arange(len(chk["ids"]))})
    sub = df.join(order, on=ID_COL, how="inner").sort("_o").drop("_o")
    p = np.column_stack([m.predict_proba(sub, mode="flat"), m.predict_proba(sub, mode="band"), m.predict_proba(sub, mode="nested")])
    return {"ok": bool(np.allclose(p, chk["proba"], atol=1e-9)), "max_abs_diff": float(np.max(np.abs(p - chk["proba"])))}


def save_regressor(m, kind: str, num: list[str], cat: list[str], bdir: Path, check_df: pl.DataFrame, extra: dict) -> dict:
    """Bundle para un CatBoost directo/residual con lista de features propia (ModelBundle asume las features V1)."""
    bdir.mkdir(parents=True, exist_ok=True)
    m.model.save_model(str(bdir / "model.cbm")); m.codec.save(bdir / "codec.json")
    man = {"model_name": m.name, "kind": kind, "strategy": STRATEGY, "features": list(m.features), "num_cols": list(num), "cat_cols": list(cat),
           "cat_feature_indices": list(m.cat_idx), "iterations": int(m.model.tree_count_), "fixed_iterations": m.fixed_iterations,
           "catboost_params": m.info.get("params"), "n_train_rows": m.info.get("n_train_rows"), "train_seconds": m.info.get("train_seconds"),
           "feature_importance": m.info.get("feature_importance"), "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "versions": env_versions(), "input_files": input_inventory(), **extra}
    if isinstance(m, CatBoostResidual):
        fb = m.fallback
        man["fallback"] = {"type": "hierarchical_median", "min_n": fb.min_n, "levels": fb.levels, "global_median": fb.global_median}
        for i, (keys, t) in enumerate(fb.tables):
            t.write_parquet(bdir / f"fallback_level{i}.parquet")
    pred = m.predict(check_df)
    np.savez_compressed(bdir / "check_predictions.npz", ids=check_df[ID_COL].to_numpy(), pred=pred.astype(np.float64))
    man["check"] = {"n_rows": int(check_df.height), "pred_sha256": hashlib.sha256(pred.astype(np.float64).tobytes()).hexdigest()}
    (bdir / "manifest.json").write_text(json.dumps(man, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return man


def load_regressor(bdir: Path):
    from catboost import CatBoostRegressor
    man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    m = make_model(man["kind"], man["num_cols"], man["cat_cols"], iterations=man["fixed_iterations"])
    assert m.features == man["features"] and m.cat_idx == man["cat_feature_indices"]
    m.codec = CategoryCodec.load(bdir / "codec.json")
    cb = CatBoostRegressor(); cb.load_model(str(bdir / "model.cbm")); m.model = cb
    if isinstance(m, CatBoostResidual):
        fbm = man["fallback"]
        fb = HierarchicalMedian(levels=fbm["levels"], min_n=fbm["min_n"]); fb.global_median = fbm["global_median"]
        fb.tables = [(keys, pl.read_parquet(bdir / f"fallback_level{i}.parquet")) for i, keys in enumerate(fbm["levels"])]
        fb.info = {"loaded": True}; m.fallback = fb
    m.info = {"loaded_from": str(bdir), "params": man.get("catboost_params")}
    return m, man


def verify_regressor(bdir: Path, df: pl.DataFrame) -> dict:
    m, man = load_regressor(bdir)
    chk = np.load(bdir / "check_predictions.npz")
    order = pl.DataFrame({ID_COL: chk["ids"], "_o": np.arange(len(chk["ids"]))})
    sub = df.join(order, on=ID_COL, how="inner").sort("_o").drop("_o")
    pred = m.predict(sub)
    return {"ok": bool(np.allclose(pred, chk["pred"], rtol=0, atol=1e-6)), "n": int(sub.height), "max_abs_diff": float(np.max(np.abs(pred - chk["pred"]))),
            "sha256_match": hashlib.sha256(pred.astype(np.float64).tobytes()).hexdigest() == man["check"]["pred_sha256"]}


# ---------------------------------------------------------------------------
# Features de ranking/training con o sin superficie
# ---------------------------------------------------------------------------
def surface_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train con estrategia aplicada, rank) con features V1 + congestión + días; RelativeCongestion ajustada con todo 2025 (sin target)."""
    from v2_features import RelativeCongestion
    train_all = load_features_v3("train", surface=True)
    train, _ = apply_outlier_strategy(train_all, STRATEGY)
    rank = load_features_v3("rank", surface=True)
    rc = RelativeCongestion().fit(train)
    return rc.transform(train), rc.transform(rank), rc.table


# ---------------------------------------------------------------------------
# train / submit
# ---------------------------------------------------------------------------
def cmd_train() -> None:
    name, proba_kind, normal_parts = finalist()
    need_surface = any(c.endswith("_s") for c in normal_parts)
    print(f"finalista {name}: proba={proba_kind} normal={normal_parts} superficie={need_surface}", flush=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {}
    pending = [c for c in normal_parts if c.endswith("_s") and not (FINAL_DIR / c / "manifest.json").exists()]   # reanudable
    if need_surface and pending:
        train_s, rank_s, rc_table = surface_frames()
        rc_table.write_parquet(FINAL_DIR / "relative_congestion_table.parquet")
        check_s = rank_s.head(50_000)
        cfg = variants()["A_surface"]
        tr_n = train_s.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
        del train_s
        for comp, kind in (("direct_normal_s", "direct_normal"), ("residual_normal_s", "residual_normal")):
            if comp not in pending:
                continue
            m = make_model(kind, cfg["num"], cfg["cat"]).fit(tr_n)
            manifests[comp] = save_regressor(m, kind, cfg["num"], cfg["cat"], FINAL_DIR / comp, check_s,
                                             {"role": comp, "train_filter": "regime_class == 0 and y <= 10800", "n_train_rows_filtered": tr_n.height})
            v = verify_regressor(FINAL_DIR / comp, rank_s); assert v["ok"] and v["sha256_match"], v
            print(f"{comp}: {manifests[comp]['iterations']} árboles, {tr_n.height:,} filas | verificación: {v}", flush=True)
            del m
        del tr_n, rank_s, check_s
        gc.collect()
    elif need_surface:
        print("bundles de superficie ya presentes (reanudación):", [c for c in normal_parts if c.endswith("_s")], flush=True)
    if proba_kind in PROBA_MODE and not (FINAL_DIR / "regime_classifier2" / "manifest.json").exists():
        train_all = load_features_v3("train", surface=False)
        train, _ = apply_outlier_strategy(train_all, STRATEGY)
        rank = load_features_v3("rank", surface=False)
        clf = RegimeClassifier2(FEATURE_NUM_COLS, FEATURE_CAT_COLS, alpha=ALPHA_PRIMARY).fit(train)
        manifests["regime_classifier2"] = save_classifier2(clf, FINAL_DIR / "regime_classifier2", rank.head(50_000))
        v = verify_classifier2(FINAL_DIR / "regime_classifier2", rank); assert v["ok"], v
        print(f"regime_classifier2: {clf.info['n_train']:,} filas, celdas={clf.info['cells_shifted_no_aobt3']} | selección anidada: {clf.loo} | verificación: {v}", flush=True)
    sel = {"selected": name, "proba": proba_kind, "normal_components": normal_parts,
           "definition": "p0*normal + p1*(MVT-SCHED) + p2*(86400+normal); normal = media simple de normal_components", "alpha": ALPHA_PRIMARY,
           "clip_negative_predictions": True, "frozen": json.loads((OUT_DIR / "finalist_frozen.json").read_text(encoding="utf-8")),
           "final_choice": json.loads((OUT_DIR / "final_choice.json").read_text(encoding="utf-8")) if (OUT_DIR / "final_choice.json").exists() else None,
           "fixed_bundles_sha256": {"residual_v1_model": _sha(V1_RESIDUAL / "model.cbm"), "v2_direct_normal_model": _sha(V2_FINAL / "direct_normal" / "model.cbm"),
                                    "v2_regime_classifier_model": _sha(V2_FINAL / "regime_classifier" / "model.cbm")},
           "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (FINAL_DIR / "selection_v3.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print("bundles V3 en", FINAL_DIR)


def predict_v3(sel: dict) -> tuple[np.ndarray, dict, dict]:
    """Predicción de ranking desde bundles (sin reentrenar). Devuelve (pred recortada, info, verificaciones)."""
    from v2_features import RelativeCongestion
    normal_parts, proba_kind = sel["normal_components"], sel["proba"]
    need_surface = any(c.endswith("_s") for c in normal_parts)
    rank = load_features_v3("rank", surface=False)
    assert rank[TARGET].null_count() == rank.height
    ver, comps = {}, {}
    if need_surface:
        train_s, rank_s, rc_table = surface_frames()
        saved = pl.read_parquet(FINAL_DIR / "relative_congestion_table.parquet")
        assert rc_table.sort("ADEP_mvt", "hour_utc").equals(saved.sort("ADEP_mvt", "hour_utc")), "la tabla de congestión relativa no coincide con la guardada"
        assert np.array_equal(rank_s[ID_COL].to_numpy(), rank[ID_COL].to_numpy())
        for comp in ("direct_normal_s", "residual_normal_s"):
            if comp in normal_parts:
                ver[comp] = verify_regressor(FINAL_DIR / comp, rank_s); assert ver[comp]["ok"] and ver[comp]["sha256_match"], ver[comp]
                m, _ = load_regressor(FINAL_DIR / comp); comps[comp] = m.predict(rank_s)
    if "direct_normal" in normal_parts:
        ver["direct_normal"] = ModelBundle.verify(V2_FINAL / "direct_normal", rank); assert ver["direct_normal"]["ok"] and ver["direct_normal"]["sha256_match"]
        m, _ = ModelBundle.load(V2_FINAL / "direct_normal"); comps["direct_normal"] = m.predict(rank)
    if "residual_v1" in normal_parts:
        ver["residual_v1"] = ModelBundle.verify(V1_RESIDUAL, rank); assert ver["residual_v1"]["ok"] and ver["residual_v1"]["sha256_match"]
        m, _ = ModelBundle.load(V1_RESIDUAL); comps["residual_v1"] = m.predict(rank)
    normal = np.mean([comps[c] for c in normal_parts], axis=0)
    if proba_kind in PROBA_MODE:
        ver["regime_classifier2"] = verify_classifier2(FINAL_DIR / "regime_classifier2", rank); assert ver["regime_classifier2"]["ok"]
        clf = load_classifier2(FINAL_DIR / "regime_classifier2")
        p = clf.predict_proba(rank, mode=PROBA_MODE[proba_kind])
        ver["regime_classifier2"]["mode"] = PROBA_MODE[proba_kind]; ver["regime_classifier2"]["band_sec_selected"] = clf.band_sec_selected
    else:
        ver["regime_classifier_v2"] = verify_classifier_v2(V2_FINAL / "regime_classifier", rank); assert ver["regime_classifier_v2"]["ok"]
        p = load_classifier_v2(V2_FINAL / "regime_classifier").predict_proba(rank)
    pred = mixture(p, normal, rank["mvt_minus_sched"].cast(pl.Float64).fill_null(0.0).to_numpy())
    info = {"n_negative_before_clip": int((pred < 0).sum()), "mean_p1": float(p[:, 1].mean()), "mean_p2": float(p[:, 2].mean()),
            "n_p1_gt_0.5": int((p[:, 1] > 0.5).sum()), "n_p2_gt_0.1": int((p[:, 2] > 0.1).sum()), "pred_max": float(pred.max()),
            "normal_components": normal_parts, "proba": proba_kind}
    return np.maximum(pred, 0.0), info, ver, rank


def cmd_submit(out: Path) -> None:
    sel = json.loads((FINAL_DIR / "selection_v3.json").read_text(encoding="utf-8"))
    for k, v in sel["fixed_bundles_sha256"].items():
        path = {"residual_v1_model": V1_RESIDUAL, "v2_direct_normal_model": V2_FINAL / "direct_normal", "v2_regime_classifier_model": V2_FINAL / "regime_classifier"}[k] / "model.cbm"
        assert _sha(path) == v, f"el bundle congelado {k} cambió"
    pred, info, ver, rank = predict_v3(sel)
    assert np.isfinite(pred).all() and pred.shape[0] == rank.height
    template = read_submitting()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}).sample(fraction=1.0, shuffle=True, seed=0), template)
    before = validate_submission(sub, template)
    out.parent.mkdir(parents=True, exist_ok=True); sub.write_parquet(out)
    re = pl.read_parquet(out); after = validate_submission(re, template); assert re.equals(sub)
    sha = _sha(out)
    v1_file, v2_file = SUBMISSIONS_DIR / "candidate_v1.parquet", SUBMISSIONS_DIR / "candidate_v2.parquet"
    meta = {"file": str(out.relative_to(PROJECT_ROOT)), "sha256": sha, "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selection": sel, "predict_info": info,
            "bundle_verification": ver, "checks_before_write": before, "checks_after_reopen": after, "uploaded_to_bucket": False,
            "candidate_v1_sha256_unchanged": _sha(v1_file) if v1_file.exists() else None,
            "candidate_v2_sha256_unchanged": _sha(v2_file) if v2_file.exists() else None}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"file": meta["file"], "rows": re.height, "sha256": sha, "checks_after_reopen": after, "predict_info": info}, indent=2, ensure_ascii=False))
    print("NO se ha subido nada al bucket del concurso.")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "submit"]); ap.add_argument("--out", default=str(SUBMISSIONS_DIR / "candidate_v3.parquet"))
    a = ap.parse_args()
    cmd_train() if a.cmd == "train" else cmd_submit(Path(a.out))


if __name__ == "__main__":
    main()
