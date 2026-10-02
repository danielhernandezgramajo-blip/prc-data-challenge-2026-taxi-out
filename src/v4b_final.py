"""
V4b — Candidata final: la configuración C2 validada (rama normal V4b + compuerta multivariable), entrenada con los 12 meses
de 2025, para el ranking (enero y julio 2026). NO sube nada.

    python src/v4b_final.py                         # 3 semillas (42, 7, 2025) como en la validación
    python src/v4b_final.py --seeds 42              # 1 semilla (más rápido; algo menos estable)
    python src/v4b_final.py --out submissions/candidate_v4b.parquet

Componentes por semilla (todas las salidas de 2025, drop_nonpositive, relativas de congestión ajustadas con 2025):
  c2_n     rama normal V4b (NormalV4b: 3000 it, depth 8, interacciones explícitas, ctr 1)
  c1_clf   compuerta de clase 1 (GateClassifier: ctr 2, negativos al 30 % con peso 1/0.3)
  c0_clf   RegimeClassifier2 de V3 (modo anidado, α = 20): solo para la celda de fecha desplazada
Predicción por semilla:
  celda (fecha MVT > fecha SCHED y sin AOBT_3):  p0·b + p1·(MVT − SCHED) + p2·(86 400 + b)  con (p0, p1, p2) de c0_clf
  resto:                                          b + p1_nuevo·(MVT − SCHED − b)
  recorte a ≥ 0; candidata = media de las semillas.
Variante de etapa 2: "S" (mezcla suave con p tal cual). La elegida en la validación fue "S_T", cuyo Platt ajustado en feb/ago
fue ≈ identidad ((a, c) ≈ (0.97, −0.03)); "S" evita reservar meses de calibración y deja entrenar con los 12 meses
(validación: S −14.8 s y S_T −14.6 s frente a V3). Las predicciones por semilla se guardan en
models/experiments/v4b/final/ (fuera de git) y la ejecución se puede reanudar.
"""
from __future__ import annotations

import argparse
import gc
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

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, PROJECT_ROOT, SUBMISSIONS_DIR  # noqa: E402
from modeling_utils import apply_outlier_strategy  # noqa: E402
from v2_protocol import DAY_SEC, STRATEGY  # noqa: E402
from v3_protocol import shifted_no_aobt3_expr  # noqa: E402
from v4b_features import EXP_DIR, load_v4b  # noqa: E402
from v4b_protocol import NCPU, SEEDS, SMOKE, SMOKE_IT, GateClassifier, NormalV4b  # noqa: E402

FINAL_DIR = (EXP_DIR / "smoke" / "final") if SMOKE else (EXP_DIR / "final")
DEFAULT_OUT = (EXP_DIR / "smoke" / "candidate_v4b_smoke.parquet") if SMOKE else (SUBMISSIONS_DIR / "candidate_v4b.parquet")
PROTECTED = ("candidate_v1.parquet", "candidate_v2.parquet", "candidate_v3.parquet")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combine(P0: np.ndarray, b: np.ndarray, p1n: np.ndarray, s: np.ndarray, in_cell: np.ndarray) -> np.ndarray:
    """Predicción de C2 (variante S): celda → mezcla de 3 clases de V3 con la normal nueva; resto → b + p1·(s − b). ≥ 0."""
    out = b + p1n * (s - b)
    cm = P0[:, 0] * b + P0[:, 1] * s + P0[:, 2] * (DAY_SEC + b)
    out = np.where(in_cell, cm, out)
    return np.maximum(out, 0.0)


def fit_seed(seed: int, train: pl.DataFrame, rank: pl.DataFrame) -> dict[str, np.ndarray]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for comp in ("c2_n", "c1_clf", "c0_clf"):
        f = FINAL_DIR / f"{comp}__s{seed}__rank.npy"
        if f.exists():
            out[comp] = np.load(f).astype(np.float64); log(f"s{seed} {comp}: en caché"); continue
        t0 = time.time(); log(f"s{seed} {comp}: entrenando …")
        if comp == "c2_n":
            m = NormalV4b(seed).fit(train); pred = m.predict(rank); info = m.info
        elif comp == "c1_clf":
            m = GateClassifier(seed).fit(train); pred = m.predict(rank); info = m.info
        else:
            import v3_protocol
            v3_protocol.SEED = seed
            m = v3_protocol.RegimeClassifier2(FEATURE_NUM_COLS, FEATURE_CAT_COLS, **({"iterations": SMOKE_IT} if SMOKE else {})).fit(train)
            pred = m.predict_proba(rank, mode="nested"); info = {k: v for k, v in m.info.items() if k != "feature_importance"}
        pred = np.asarray(pred, dtype=np.float64)
        assert np.isfinite(pred).all() and pred.shape[0] == rank.height
        np.save(f, pred.astype(np.float32))
        (FINAL_DIR / f"{comp}__s{seed}__info.json").write_text(json.dumps({k: v for k, v in info.items() if k != "feature_importance"},
                                                                           default=str, indent=1), encoding="utf-8")
        out[comp] = pred.astype(np.float32).astype(np.float64)
        log(f"s{seed} {comp}: hecho en {(time.time() - t0) / 60:.1f} min")
        del m; gc.collect()
    return out


def main() -> None:
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    from v2_features import RelativeCongestion
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    t0 = time.time()
    protected = {n: sha256(SUBMISSIONS_DIR / n) for n in PROTECTED if (SUBMISSIONS_DIR / n).exists()}
    log(f"CPU: {NCPU} hilos | semillas: {args.seeds}{' | SMOKE' if SMOKE else ''}")
    train = load_v4b("train"); rank = load_v4b("rank")
    if SMOKE:
        train = train.sample(fraction=0.03, seed=0).sort("MVT_TIME_UTC_mvt")
    assert train["MVT_TIME_UTC_mvt"].dt.year().unique().to_list() == [2025]
    assert rank["TAXITIME_SEC_mvt"].null_count() == rank.height, "el target del ranking debe estar oculto"
    train, _ = apply_outlier_strategy(train, STRATEGY)
    rc = RelativeCongestion().fit(train)
    train, rank = rc.transform(train), rc.transform(rank)
    log(f"train {train.height:,} filas (12 meses) | ranking {rank.height:,} salidas")
    s = rank["mvt_minus_sched"].cast(pl.Float64).to_numpy()
    in_cell = rank.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy()
    preds, comps = [], {}
    for seed in args.seeds:
        c = fit_seed(seed, train, rank)
        preds.append(combine(c["c0_clf"], c["c2_n"], c["c1_clf"], s, in_cell))
        comps[seed] = c
    pred = np.mean(preds, axis=0)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    template = read_submitting()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}), template)
    before = validate_submission(sub, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"{out_path} ya existe: no se sobrescribe (bórrala o usa --out)")
    sub.write_parquet(out_path)
    re = pl.read_parquet(out_path); after = validate_submission(re, template); assert re.equals(sub)
    sha = sha256(out_path)
    for n, h in protected.items():
        assert sha256(SUBMISSIONS_DIR / n) == h, f"{n} cambió durante la ejecución"
    p1 = np.mean([c["c1_clf"] for c in comps.values()], axis=0)
    info = {"n_in_cell": int(in_cell.sum()), "mean_p1_new": float(p1.mean()), "n_p1_gt_0.5": int((p1 > 0.5).sum()),
            "pred_mean": float(pred.mean()), "pred_p99": float(np.quantile(pred, 0.99)), "pred_max": float(pred.max()),
            "seed_spread_mean_abs": float(np.mean(np.abs(preds[0] - pred))) if len(preds) > 1 else 0.0}
    meta = {"file": str(out_path.relative_to(PROJECT_ROOT)) if out_path.is_relative_to(PROJECT_ROOT) else str(out_path),
            "sha256": sha, "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "V4b C2 (rama normal V4b + compuerta multivariable; variante S); 12 meses de 2025",
            "validation": "outputs/experiments/v4b/report.md (C2 −14.6 s vs V3 en ene/jul 2025; no superó el umbral de −15 s de la etapa costosa)",
            "seeds": args.seeds, "smoke": SMOKE, "threads": NCPU, "predict_info": info, "checks_before_write": before,
            "checks_after_reopen": after, "protected_unchanged": {n: h for n, h in protected.items()},
            "uploaded_to_bucket": False, "minutes": round((time.time() - t0) / 60, 1)}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(json.dumps({"file": meta["file"], "rows": re.height, "sha256": sha, "ok": after["ok"], "predict_info": info,
                    "minutes": meta["minutes"]}, ensure_ascii=False))
    log("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
