"""
Genera una submission candidata a partir de los bundles finales (sin reentrenar).

Pasos:
 1. Lee models/final/selection.json (modelo/ensamble elegido por select_model.py y si se recortan negativos).
 2. Carga los bundles (ModelBundle.load) y verifica que la predicción tras cargar == la guardada al entrenar.
 3. Construye las features de ranking con la MISMA función que en training (build_features).
 4. Predice; aplica el ensamble 50/50 si fue el seleccionado; recorta negativos a 0 si los backtests lo avalan.
 5. Une por MVT_ID_mvt (no por posición) con submitting.parquet y restaura EXACTAMENTE su orden.
 6. Preserva el esquema del template (MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32).
 7. Valida: 344 841 filas, IDs únicos, ni faltantes ni extra, cero nulos, valores finitos, esquema idéntico.
 8. Escribe submissions/candidate_v1.parquet, lo vuelve a abrir y repite TODAS las validaciones.

NO sube nada al bucket del concurso.
Uso: python src/make_submission.py [--out submissions/candidate_v1.parquet]
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

from config import EXPECTED_SUBMITTING_ROWS, ID_COL, MODELS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR, TARGET  # noqa: E402
from io_utils import read_submitting  # noqa: E402
from prepare_training import load_features  # noqa: E402
from train_baseline import ModelBundle  # noqa: E402


def assemble_submission(pred_df: pl.DataFrame, template: pl.DataFrame) -> pl.DataFrame:
    """pred_df: columnas [MVT_ID_mvt, pred (float)]. Devuelve un frame con el esquema y el ORDEN del template.

    La unión es por MVT_ID_mvt; el orden del template se restaura con un índice explícito, así que
    el resultado no depende del orden en que lleguen las predicciones."""
    tmpl = template.select(ID_COL).with_row_index("_pos")
    joined = tmpl.join(pred_df.select(ID_COL, "pred"), on=ID_COL, how="left").sort("_pos")
    if joined["pred"].null_count():
        raise ValueError(f"{joined['pred'].null_count()} IDs del template sin predicción")
    target_dtype = template.schema[TARGET]
    out = joined.select(
        pl.col(ID_COL).cast(template.schema[ID_COL]),
        pl.col("pred").round(0).cast(target_dtype).alias(TARGET),
    )
    return out


def validate_submission(sub: pl.DataFrame, template: pl.DataFrame) -> dict:
    """Valida y devuelve un dict de comprobaciones; lanza AssertionError si alguna falla."""
    checks = {}
    checks["n_rows_ok"] = sub.height == template.height == EXPECTED_SUBMITTING_ROWS
    checks["schema_ok"] = list(sub.schema.items()) == list(template.schema.items())
    checks["ids_unique"] = sub[ID_COL].n_unique() == sub.height
    ids_sub, ids_tmpl = set(sub[ID_COL].to_list()), set(template[ID_COL].to_list())
    checks["ids_missing"] = len(ids_tmpl - ids_sub)
    checks["ids_extra"] = len(ids_sub - ids_tmpl)
    checks["order_identical"] = bool((sub[ID_COL] == template[ID_COL]).all()) if sub.height == template.height else False
    checks["nulls_target"] = int(sub[TARGET].null_count())
    checks["nulls_id"] = int(sub[ID_COL].null_count())
    vals = sub[TARGET].cast(pl.Float64).to_numpy()
    checks["all_finite"] = bool(np.isfinite(vals).all())
    checks["min"] = float(np.min(vals)); checks["max"] = float(np.max(vals)); checks["mean"] = float(np.mean(vals))
    checks["n_negative"] = int((vals < 0).sum())
    ok = (checks["n_rows_ok"] and checks["schema_ok"] and checks["ids_unique"] and checks["ids_missing"] == 0
          and checks["ids_extra"] == 0 and checks["order_identical"] and checks["nulls_target"] == 0
          and checks["nulls_id"] == 0 and checks["all_finite"])
    checks["ok"] = bool(ok)
    assert ok, f"validación de la submission fallida: {checks}"
    return checks


def predict_ranking(selection: dict, rank: pl.DataFrame) -> tuple[np.ndarray, dict]:
    info = {}
    preds = {}
    needed = ["catboost_direct", "catboost_residual"] if selection["selected"] == "ensemble_50_50" else [selection["selected"]]
    for name in needed:
        bdir = PROJECT_ROOT / selection["bundles"][name]
        ver = ModelBundle.verify(bdir, rank)
        assert ver["ok"] and ver["sha256_match"], f"bundle {name} no reproduce sus predicciones de verificación: {ver}"
        m, manifest = ModelBundle.load(bdir)
        preds[name] = m.predict(rank)
        info[name] = {"bundle": str(bdir.relative_to(PROJECT_ROOT)), "iterations": manifest["iterations"], "strategy": manifest["strategy"],
                      "trained_at_utc": manifest["trained_at_utc"], "verify": ver,
                      "pct_fallback": m.info.get("pct_pred_fallback")}
    if selection["selected"] == "ensemble_50_50":
        w = selection["ensemble_weight_direct"]
        p = w * preds["catboost_direct"] + (1 - w) * preds["catboost_residual"]
    else:
        p = preds[selection["selected"]]
    info["n_negative_before_clip"] = int((p < 0).sum())
    if selection.get("clip_negative_predictions", False):
        p = np.maximum(p, 0.0)
    info["clipped"] = bool(selection.get("clip_negative_predictions", False))
    assert np.isfinite(p).all(), "predicciones no finitas"
    return p, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SUBMISSIONS_DIR / "candidate_v1.parquet"))
    args = ap.parse_args()
    out = Path(args.out)
    selection = json.loads((MODELS_DIR / "final" / "selection.json").read_text(encoding="utf-8"))
    print("Selección:", selection["selected"], "| estrategia:", selection["strategy"], "| clip negativos:", selection["clip_negative_predictions"])

    rank = load_features("rank")
    assert rank[TARGET].null_count() == rank.height, "el ranking DEP no debe traer target"
    template = read_submitting()
    p, info = predict_ranking(selection, rank)
    pred_df = pl.DataFrame({ID_COL: rank[ID_COL], "pred": p})
    # Prueba de robustez: el ensamblado no depende del orden de llegada de las predicciones
    sub = assemble_submission(pred_df.sample(fraction=1.0, shuffle=True, seed=0), template)
    checks_before = validate_submission(sub, template)

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(out)
    reopened = pl.read_parquet(out)
    checks_after = validate_submission(reopened, template)
    assert reopened.equals(sub), "el archivo reabierto difiere del generado"
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"file": str(out.relative_to(PROJECT_ROOT)) if out.is_relative_to(PROJECT_ROOT) else str(out), "sha256": sha,
            "rows": reopened.height, "schema": {k: str(v) for k, v in reopened.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selection": selection, "models": info, "checks_before_write": checks_before, "checks_after_reopen": checks_after,
            "uploaded_to_bucket": False}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"file": meta["file"], "rows": meta["rows"], "sha256": sha, "checks_after_reopen": checks_after}, indent=2, ensure_ascii=False))
    print("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
