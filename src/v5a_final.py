"""
V5a — Candidata: V4b C2 con la rama normal mezclada CatBoost + LightGBM (b = 0.5·c2_n + 0.5·lgb_const), misma compuerta y
misma celda de fecha desplazada que V4b (variante S). Entrenada con los 12 meses de 2025. NO sube nada.

    python src/v5a_final.py                       # requiere las predicciones por semilla de V4b en models/experiments/v4b/final/

Validación (outputs/experiments/v5/protocol.md §2, outer_check.json): −1.41 s de RMSE_w frente a C2 en ene/jul 2025
(IC 95 % [−2.01, −0.87]; enero −0.80, julio −1.94; no depende de filas sueltas). La compuerta mezclada (§2b) NO se usa:
su ganancia venía de una sola fila por mes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import ID_COL, MODELS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR  # noqa: E402
from modeling_utils import apply_outlier_strategy  # noqa: E402
from v2_protocol import STRATEGY  # noqa: E402
from v3_protocol import shifted_no_aobt3_expr  # noqa: E402
from v4b_features import load_v4b  # noqa: E402
from v4b_final import combine  # noqa: E402

V4B_FINAL = MODELS_DIR / "experiments" / "v4b" / "final"
V5_FINAL = MODELS_DIR / "experiments" / "v5" / "final"
DEFAULT_OUT = SUBMISSIONS_DIR / "candidate_v5a.parquet"
PROTECTED = ("candidate_v1.parquet", "candidate_v2.parquet", "candidate_v3.parquet", "candidate_v4b.parquet")
LGB_SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    from v2_features import RelativeCongestion
    from v5_lgb import LGBNormal
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2025], help="semillas de V4b ya calculadas")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lgb-iterations", type=int, default=2000, help="solo para pruebas; la validación usó 2000")
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} ya existe: no se sobrescribe (bórrala o usa --out)")
    t0 = time.time()
    need = [V4B_FINAL / f"{c}__s{s}__rank.npy" for s in args.seeds for c in ("c2_n", "c1_clf", "c0_clf")]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Faltan predicciones de V4b ({missing}); ejecuta antes src/v4b_run_candidate.sh")
    protected = {n: sha256(SUBMISSIONS_DIR / n) for n in PROTECTED if (SUBMISSIONS_DIR / n).exists()}
    rank = load_v4b("rank")
    assert rank["TAXITIME_SEC_mvt"].null_count() == rank.height, "el target del ranking debe estar oculto"
    tag = "" if args.lgb_iterations == 2000 else f"__it{args.lgb_iterations}"
    lgb_path = V5_FINAL / f"lgb_const__s{LGB_SEED}{tag}__rank.npy"
    V5_FINAL.mkdir(parents=True, exist_ok=True)
    if lgb_path.exists():
        lgb = np.load(lgb_path).astype(np.float64); log("lgb_const: en caché"); info = {}
    else:
        train = load_v4b("train")
        assert train["MVT_TIME_UTC_mvt"].dt.year().unique().to_list() == [2025]
        train, _ = apply_outlier_strategy(train, STRATEGY)
        rc = RelativeCongestion().fit(train)
        train, rank = rc.transform(train), rc.transform(rank)
        log(f"train {train.height:,} filas (12 meses) | ranking {rank.height:,}; entrenando LightGBM …")
        m = LGBNormal(LGB_SEED, "const", iterations=args.lgb_iterations).fit(train)
        lgb = m.predict(rank); info = {k: v for k, v in m.info.items() if k != "top_gain"}
        assert np.isfinite(lgb).all() and lgb.shape[0] == rank.height
        np.save(lgb_path, lgb.astype(np.float32)); lgb = lgb.astype(np.float32).astype(np.float64)
        (V5_FINAL / f"lgb_const__s{LGB_SEED}{tag}__info.json").write_text(json.dumps(m.info, default=str, indent=1), encoding="utf-8")
        log(f"lgb_const hecho en {(time.time() - t0) / 60:.1f} min")
        del train, m
    s = rank["mvt_minus_sched"].cast(pl.Float64).to_numpy()
    in_cell = rank.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy()
    preds = []
    for sd in args.seeds:
        c2n = np.load(V4B_FINAL / f"c2_n__s{sd}__rank.npy").astype(np.float64)
        p1 = np.load(V4B_FINAL / f"c1_clf__s{sd}__rank.npy").astype(np.float64)
        P0 = np.load(V4B_FINAL / f"c0_clf__s{sd}__rank.npy").astype(np.float64)
        assert c2n.shape[0] == p1.shape[0] == P0.shape[0] == rank.height
        preds.append(combine(P0, 0.5 * c2n + 0.5 * lgb, p1, s, in_cell))
    pred = np.mean(preds, axis=0)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    template = read_submitting()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}), template)
    before = validate_submission(sub, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(out_path)
    re = pl.read_parquet(out_path); after = validate_submission(re, template); assert re.equals(sub)
    for n, h in protected.items():
        assert sha256(SUBMISSIONS_DIR / n) == h, f"{n} cambió durante la ejecución"
    v4b = SUBMISSIONS_DIR / "candidate_v4b.parquet"
    diff = None
    if v4b.exists():
        old = pl.read_parquet(v4b)
        j = re.join(old, on=ID_COL, suffix="_v4b")
        cols = [c for c in re.columns if c != ID_COL]
        if cols:
            d = (j[cols[0]] - j[f"{cols[0]}_v4b"]).abs().to_numpy()
            diff = {"mean_abs_vs_v4b": float(d.mean()), "p99_abs_vs_v4b": float(np.quantile(d, 0.99)), "max_abs_vs_v4b": float(d.max())}
    meta = {"file": str(out_path.relative_to(PROJECT_ROOT)) if out_path.is_relative_to(PROJECT_ROOT) else str(out_path),
            "sha256": sha256(out_path), "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "V5a = V4b C2 (variante S) con rama normal 0.5·CatBoost c2_n + 0.5·LightGBM (hojas constantes); 12 meses de 2025",
            "validation": "outputs/experiments/v5/outer_check.json (−1.41 s RMSE_w vs C2 en ene/jul 2025)",
            "seeds_v4b": args.seeds, "lgb_seed": LGB_SEED, "lgb_info": info, "diff_vs_v4b": diff,
            "checks_before_write": before, "checks_after_reopen": after, "protected_unchanged": protected,
            "uploaded_to_bucket": False, "minutes": round((time.time() - t0) / 60, 1)}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(json.dumps({"file": meta["file"], "rows": re.height, "sha256": meta["sha256"], "ok": after["ok"], "diff_vs_v4b": diff},
                   ensure_ascii=False))
    log("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
