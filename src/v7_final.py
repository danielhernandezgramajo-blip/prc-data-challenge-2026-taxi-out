"""
V7 — Candidata: V6 + corrección de las filas SIN AOBT_3 propio en congestión (outputs/experiments/v7/protocol.md, V7c):
en filas con has_aobt3 = 0, fuera de la celda, sin regla LOBT y anomalía de vecinos nba > 600 s, la rama normal pasa a
b' = max(b, nb), con nb = media de MVT − AOBT_3 de las demás salidas del aeropuerto en ±60 min (solo features del ranking).
La mediana de referencia de nb por aeropuerto sale de 2025. NO sube nada.

    python src/v7_final.py
"""
from __future__ import annotations

import argparse
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

from config import ID_COL, MODELS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR, TARGET  # noqa: E402
from v6_final import LGB_SEED, V4B_FINAL, V5_FINAL, apply_lobt_rule, fit_lobt_rule, log, sha256  # noqa: E402
from v7_congestion import airport_baseline, anomaly, neighbour_congestion  # noqa: E402
from v7c_noaobt import THRESHOLD, apply_max_nb, noaobt_mask  # noqa: E402

DEFAULT_OUT = SUBMISSIONS_DIR / "candidate_v7.parquet"
PROTECTED = ("candidate_v1.parquet", "candidate_v2.parquet", "candidate_v3.parquet", "candidate_v4b.parquet",
             "candidate_v5a.parquet", "candidate_v6.parquet")


def main() -> None:
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    from v3_protocol import shifted_no_aobt3_expr
    from v4b_features import load_v4b
    from v4b_final import combine
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2025])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} ya existe: no se sobrescribe (bórrala o usa --out)")
    t0 = time.time()
    lgb_path = V5_FINAL / f"lgb_const__s{LGB_SEED}__rank.npy"
    need = [V4B_FINAL / f"{c}__s{s}__rank.npy" for s in args.seeds for c in ("c2_n", "c1_clf", "c0_clf")] + [lgb_path]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Faltan {missing} (V4b: src/v4b_run_candidate.sh; LightGBM: src/v5a_final.py)")
    protected = {n: sha256(SUBMISSIONS_DIR / n) for n in PROTECTED if (SUBMISSIONS_DIR / n).exists()}

    rank = load_v4b("rank")
    assert rank[TARGET].null_count() == rank.height, "el target del ranking debe estar oculto"
    train = load_v4b("train")
    rule = fit_lobt_rule(train)
    assert 0.9 < rule["c"] < 1.1 and rule["n"] >= 100, rule
    nb_tr = neighbour_congestion(train); base = airport_baseline(train, nb_tr)
    del train
    nb = neighbour_congestion(rank); nba = anomaly(rank, nb, base)
    s = rank["mvt_minus_sched"].cast(pl.Float64).to_numpy()
    in_cell = rank.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy()
    L = rank["mvt_minus_lobt"].cast(pl.Float64).to_numpy()
    has = rank["has_aobt3"].to_numpy()
    mask = noaobt_mask(nba, has, L, in_cell, nb, THRESHOLD)
    log(f"regla LOBT: a={rule['a']:.1f}, c={rule['c']:.4f} | filas V7c (sin AOBT_3, nba > {THRESHOLD:.0f}): {int(mask.sum())}")
    lgb = np.load(lgb_path).astype(np.float64)
    v6, v7 = [], []
    for sd in args.seeds:
        b = 0.5 * np.load(V4B_FINAL / f"c2_n__s{sd}__rank.npy").astype(np.float64) + 0.5 * lgb
        p1 = np.load(V4B_FINAL / f"c1_clf__s{sd}__rank.npy").astype(np.float64)
        P0 = np.load(V4B_FINAL / f"c0_clf__s{sd}__rank.npy").astype(np.float64)
        v6.append(combine(P0, b, p1, s, in_cell))
        v7.append(combine(P0, apply_max_nb(b, nb, mask), p1, s, in_cell))
    base6 = apply_lobt_rule(np.mean(v6, axis=0), L, rule)
    pred = apply_lobt_rule(np.mean(v7, axis=0), L, rule)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    d = pred - base6
    adep = rank["ADEP_mvt"].to_numpy()
    chg = np.abs(d) > 0.5
    info = {"rows_masked": int(mask.sum()), "rows_changed": int(chg.sum()), "mean_raise_sec": float(d[chg].mean()) if chg.any() else 0.0,
            "median_raise_sec": float(np.median(d[chg])) if chg.any() else 0.0, "max_raise_sec": float(d.max()),
            "by_airport": {k: int(v) for k, v in zip(*np.unique(adep[chg], return_counts=True))}, "threshold_nba": THRESHOLD}
    template = read_submitting()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}), template)
    before = validate_submission(sub, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(out_path)
    re = pl.read_parquet(out_path); after = validate_submission(re, template); assert re.equals(sub)
    for n, h in protected.items():
        assert sha256(SUBMISSIONS_DIR / n) == h, f"{n} cambió durante la ejecución"
    diffs = {}
    v6p = SUBMISSIONS_DIR / "candidate_v6.parquet"
    if v6p.exists():
        j = re.join(pl.read_parquet(v6p), on=ID_COL, suffix="_v6")
        dd = (j[TARGET] - j[f"{TARGET}_v6"]).to_numpy()
        diffs["candidate_v6.parquet"] = {"rows_changed": int((dd != 0).sum()), "mean_abs": float(np.abs(dd).mean()),
                                         "max_abs": float(np.abs(dd).max()), "rows_lowered": int((dd < 0).sum())}
    meta = {"file": str(out_path.relative_to(PROJECT_ROOT)) if out_path.is_relative_to(PROJECT_ROOT) else str(out_path),
            "sha256": sha256(out_path), "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "V7 = V6 + filas sin AOBT_3 propio en congestión (nba > 600 s): rama normal max(b, nb)",
            "validation": {"v7c": "outputs/experiments/v7/noaobt_check.json (ΔRMSE_w −0.35 s, IC95 [−0.86, −0.01]; ene −0.74, feb −8.84)",
                           "lobt_rule": "outputs/experiments/v6/rule_check.json", "v5a_blend": "outputs/experiments/v5/outer_check.json"},
            "lobt_rule": rule, "v7c": info, "seeds_v4b": args.seeds, "lgb_cache_sha256": sha256(lgb_path), "diffs": diffs,
            "checks_before_write": before, "checks_after_reopen": after, "protected_unchanged": protected,
            "uploaded_to_bucket": False, "minutes": round((time.time() - t0) / 60, 1)}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(json.dumps({"file": meta["file"], "sha256": meta["sha256"], "ok": after["ok"], "v7c": info, "diffs": diffs}, ensure_ascii=False))
    log("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
