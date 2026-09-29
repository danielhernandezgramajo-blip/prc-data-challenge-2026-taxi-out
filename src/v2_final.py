"""
V2 — entrenamiento final de la variante S_mix con todo 2025 y generación de submissions/candidate_v2.parquet.

    python src/v2_final.py train      # bundles en models/experiments/v2/final/ (no toca models/final de V1)
    python src/v2_final.py submit     # candidate_v2.parquet desde los bundles, sin reentrenar; validación V1-idéntica

Definición congelada (outputs/experiments/v2/finalist_frozen.json):
    pred = p0·normal + p1·(MVT − SCHED) + p2·(86 400 + normal),  normal = 0.5·direct_normal + 0.5·residual_V1
    p    = RegimeClassifier (CatBoost binario clase 1 + tasa suavizada clase 2), features V1, todo 2025 (drop_nonpositive)
    direct_normal = CatBoost directo (config V1, 1000 it.) con filas de clase 0 y y ≤ 3 h de todo 2025
    residual_V1   = models/final/catboost_residual (bundle V1, sin cambios; se verifica su sha256)
    negativos → 0 (como V1). Sin suelo AOBT_3.
"""
from __future__ import annotations

import argparse
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
from prepare_training import load_features  # noqa: E402
from train_baseline import ModelBundle, env_versions, input_inventory  # noqa: E402
from v2_protocol import DAY_SEC, STRATEGY, RegimeClassifier, make_model, regime_class_expr  # noqa: E402

FINAL_DIR = MODELS_DIR / "experiments" / "v2" / "final"
V1_RESIDUAL = MODELS_DIR / "final" / "catboost_residual"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_classifier(m: RegimeClassifier, bdir: Path, check_df: pl.DataFrame) -> dict:
    bdir.mkdir(parents=True, exist_ok=True)
    m.clf.save_model(str(bdir / "model.cbm"))
    m.codec.save(bdir / "codec.json")
    (bdir / "rate2.json").write_text(json.dumps({"global": m.rate2_global, "table": [{"ADEP_mvt": k[0], "has_aobt3": k[1], "rate": v} for k, v in m.rate2.items()]}, indent=1), encoding="utf-8")
    p = m.predict_proba(check_df)
    np.savez_compressed(bdir / "check_predictions.npz", ids=check_df[ID_COL].to_numpy(), proba=p)
    man = {"model_name": "regime_classifier", "features": m.num + m.cat, "cat_cols": m.cat, "iterations": m.iterations,
           "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": env_versions(),
           "input_files": input_inventory(), "info": m.info, "check": {"n_rows": check_df.height, "proba_sha256": hashlib.sha256(p.tobytes()).hexdigest()}}
    (bdir / "manifest.json").write_text(json.dumps(man, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return man


def load_classifier(bdir: Path) -> RegimeClassifier:
    from catboost import CatBoostClassifier
    man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    m = RegimeClassifier(FEATURE_NUM_COLS, FEATURE_CAT_COLS, iterations=man["iterations"])
    assert m.num + m.cat == man["features"]
    m.codec = CategoryCodec.load(bdir / "codec.json")
    m.clf = CatBoostClassifier(); m.clf.load_model(str(bdir / "model.cbm"))
    r = json.loads((bdir / "rate2.json").read_text(encoding="utf-8"))
    m.rate2 = {(t["ADEP_mvt"], int(t["has_aobt3"])): t["rate"] for t in r["table"]}; m.rate2_global = r["global"]
    return m


def verify_classifier(bdir: Path, df: pl.DataFrame) -> dict:
    m = load_classifier(bdir)
    chk = np.load(bdir / "check_predictions.npz")
    order = pl.DataFrame({ID_COL: chk["ids"], "_o": np.arange(len(chk["ids"]))})
    sub = df.join(order, on=ID_COL, how="inner").sort("_o").drop("_o")
    p = m.predict_proba(sub)
    return {"ok": bool(np.allclose(p, chk["proba"], atol=1e-9)), "max_abs_diff": float(np.max(np.abs(p - chk["proba"])))}


def predict_v2(rank: pl.DataFrame) -> tuple[np.ndarray, dict]:
    clf = load_classifier(FINAL_DIR / "regime_classifier")
    dn, man_dn = ModelBundle.load(FINAL_DIR / "direct_normal")
    rs, man_rs = ModelBundle.load(V1_RESIDUAL)
    p = clf.predict_proba(rank)
    normal = 0.5 * dn.predict(rank) + 0.5 * rs.predict(rank)
    sched = rank["mvt_minus_sched"].cast(pl.Float64).fill_null(0.0).to_numpy()
    pred = p[:, 0] * normal + p[:, 1] * sched + p[:, 2] * (DAY_SEC + normal)
    info = {"n_negative_before_clip": int((pred < 0).sum()), "mean_p1": float(p[:, 1].mean()), "mean_p2": float(p[:, 2].mean()),
            "n_p1_gt_0.5": int((p[:, 1] > 0.5).sum()), "direct_normal_trained_at": man_dn["trained_at_utc"], "residual_v1_trained_at": man_rs["trained_at_utc"]}
    return np.maximum(pred, 0.0), info


def cmd_train() -> None:
    train_all = load_features("train")
    train, sinfo = apply_outlier_strategy(train_all, STRATEGY)
    rank = load_features("rank"); check = rank.head(50_000)
    print(f"train {train.height:,} filas ({STRATEGY})", flush=True)
    clf = RegimeClassifier(FEATURE_NUM_COLS, FEATURE_CAT_COLS).fit(train)
    man_c = save_classifier(clf, FINAL_DIR / "regime_classifier", check)
    v = verify_classifier(FINAL_DIR / "regime_classifier", rank); assert v["ok"], v
    print(f"regime_classifier: {clf.info} | verificación carga: {v}", flush=True)
    tr_n = train.filter((regime_class_expr() == 0) & (pl.col(TARGET) <= OUTLIER_HI_SEC))
    dn = make_model("direct_normal", FEATURE_NUM_COLS, FEATURE_CAT_COLS).fit(tr_n)
    dn.name = "catboost_direct_normal"
    man_d = ModelBundle(dn, STRATEGY, extra={"train_filter": "regime_class == 0 and y <= 10800", "n_train_rows_filtered": tr_n.height}).save(FINAL_DIR / "direct_normal", check)
    vd = ModelBundle.verify(FINAL_DIR / "direct_normal", rank); assert vd["ok"] and vd["sha256_match"], vd
    print(f"direct_normal: {man_d['iterations']} árboles, {tr_n.height:,} filas | verificación: {vd}", flush=True)
    vr = ModelBundle.verify(V1_RESIDUAL, rank); assert vr["ok"] and vr["sha256_match"], vr
    sel = {"selected": "S_mix", "definition": "p0*normal + p1*(MVT-SCHED) + p2*(86400+normal); normal = 0.5*direct_normal + 0.5*residual_V1",
           "clip_negative_predictions": True, "components": {"regime_classifier": str((FINAL_DIR / "regime_classifier").relative_to(PROJECT_ROOT)),
           "direct_normal": str((FINAL_DIR / "direct_normal").relative_to(PROJECT_ROOT)), "residual_v1": str(V1_RESIDUAL.relative_to(PROJECT_ROOT))},
           "residual_v1_model_sha256": _sha(V1_RESIDUAL / "model.cbm"), "frozen": json.loads((PROJECT_ROOT / "outputs" / "experiments" / "v2" / "finalist_frozen.json").read_text()),
           "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (FINAL_DIR / "selection_v2.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print("bundles V2 en", FINAL_DIR)


def cmd_submit(out: Path) -> None:
    rank = load_features("rank")
    assert rank[TARGET].null_count() == rank.height
    template = read_submitting()
    sel = json.loads((FINAL_DIR / "selection_v2.json").read_text(encoding="utf-8"))
    assert _sha(V1_RESIDUAL / "model.cbm") == sel["residual_v1_model_sha256"], "el bundle residual V1 cambió"
    v1 = verify_classifier(FINAL_DIR / "regime_classifier", rank); v2 = ModelBundle.verify(FINAL_DIR / "direct_normal", rank); v3 = ModelBundle.verify(V1_RESIDUAL, rank)
    assert v1["ok"] and v2["ok"] and v2["sha256_match"] and v3["ok"] and v3["sha256_match"], (v1, v2, v3)
    pred, info = predict_v2(rank)
    assert np.isfinite(pred).all()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}).sample(fraction=1.0, shuffle=True, seed=0), template)
    before = validate_submission(sub, template)
    out.parent.mkdir(parents=True, exist_ok=True); sub.write_parquet(out)
    re = pl.read_parquet(out); after = validate_submission(re, template); assert re.equals(sub)
    sha = _sha(out)
    v1_file = SUBMISSIONS_DIR / "candidate_v1.parquet"
    meta = {"file": str(out.relative_to(PROJECT_ROOT)), "sha256": sha, "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selection": sel, "predict_info": info,
            "bundle_verification": {"regime_classifier": v1, "direct_normal": v2, "residual_v1": v3},
            "checks_before_write": before, "checks_after_reopen": after, "uploaded_to_bucket": False,
            "candidate_v1_sha256_unchanged": _sha(v1_file) if v1_file.exists() else None}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"file": meta["file"], "rows": re.height, "sha256": sha, "checks_after_reopen": after, "predict_info": info}, indent=2, ensure_ascii=False))
    print("NO se ha subido nada al bucket del concurso.")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "submit"]); ap.add_argument("--out", default=str(SUBMISSIONS_DIR / "candidate_v2.parquet"))
    a = ap.parse_args()
    cmd_train() if a.cmd == "train" else cmd_submit(Path(a.out))


if __name__ == "__main__":
    main()
