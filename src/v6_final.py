"""
V6 — Candidata: V5a (V4b C2 con rama normal 0.5·CatBoost c2_n + 0.5·LightGBM) + regla de hueco LOBT
(outputs/experiments/v6/protocol.md §1) y, solo si pasó su comprobación prerregistrada, la compuerta por prefijo de
aerolínea en las salidas de LIRF sin plan NM (§2). Entrenada con los 12 meses de 2025. NO sube nada.

    python src/v6_final.py                        # requiere models/experiments/v4b/final/*__rank.npy y la caché LightGBM de V5a
    python src/v6_final.py --prefix-gate off      # fuerza la variante sin compuerta por prefijo

Regla LOBT: filas con MVT − LOBT > 2 h → ŷ = max(0, a + c·(MVT − LOBT)), (a, c) por mínimos cuadrados en todas las filas de
2025 con MVT − LOBT > 2 h e y > 0 (143 filas; a ≈ −1228, c ≈ 1.015). Validación fuera de mes: −11.5 s RMSE_w en ene/jul
2025 (IC 95 % [−34, −0.3]); mejora también en feb/ago. La ganancia en el ranking depende sobre todo de UNA fila (24 h).
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

from config import ID_COL, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR, TARGET  # noqa: E402

LOBT_GAP_SEC = 7200.0
V4B_FINAL = MODELS_DIR / "experiments" / "v4b" / "final"
V5_FINAL = MODELS_DIR / "experiments" / "v5" / "final"
DEFAULT_OUT = SUBMISSIONS_DIR / "candidate_v6.parquet"
PREFIX_CHECK = OUTPUTS_DIR / "experiments" / "v6" / "prefix_gate_check.json"
PROTECTED = ("candidate_v1.parquet", "candidate_v2.parquet", "candidate_v3.parquet", "candidate_v4b.parquet",
             "candidate_v5a.parquet")
LGB_SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_lobt_rule(train: pl.DataFrame, threshold: float = LOBT_GAP_SEC, col: str = "mvt_minus_lobt") -> dict:
    """Recta y ≈ a + c·(MVT − LOBT) en las filas con MVT − LOBT > umbral (2 h) e y > 0 (solo filas con target conocido).
    `threshold` y `col` solo se cambian en la comprobación de sensibilidad (src/v6_rule_check.py)."""
    L = pl.col(col).cast(pl.Float64)
    # en Polars NaN > x es verdadero: se excluyen NaN y nulos explícitamente
    t = train.filter(L.is_not_null() & L.is_not_nan() & (L > threshold) & (pl.col(TARGET).cast(pl.Float64) > 0))
    x = t[col].cast(pl.Float64).to_numpy()
    y = t[TARGET].cast(pl.Float64).to_numpy()
    if x.size < 10 or np.ptp(x) <= 0:
        raise ValueError(f"muy pocas filas para ajustar la regla LOBT ({x.size})")
    c, a = np.polyfit(x, y, 1)
    resid = y - (a + c * x)
    return {"a": float(a), "c": float(c), "n": int(x.size), "threshold_sec": float(threshold), "column": col,
            "rmse_in_sample": float(np.sqrt(np.mean(resid ** 2))), "share_y_gt_1h": float(np.mean(y > 3600))}


def apply_lobt_rule(pred: np.ndarray, L: np.ndarray, rule: dict) -> np.ndarray:
    """Sustituye la predicción por max(0, a + c·L) donde L = MVT − LOBT > umbral; L nulo/NaN no se toca."""
    L = np.asarray(L, dtype=np.float64)
    m = np.isfinite(L) & (L > rule["threshold_sec"])
    out = np.asarray(pred, dtype=np.float64).copy()
    out[m] = np.maximum(rule["a"] + rule["c"] * L[m], 0.0)
    return out


def prefix_gate_enabled(mode: str) -> tuple[bool, str]:
    if mode in ("on", "off"):
        return mode == "on", f"forzado por --prefix-gate {mode}"
    if not PREFIX_CHECK.exists():
        return False, "sin comprobación (prefix_gate_check.json no existe) → desactivada"
    chk = json.loads(PREFIX_CHECK.read_text(encoding="utf-8"))
    return bool(chk.get("pass")), f"regla prerregistrada: pass={chk.get('pass')} (ΔRMSE_w {chk.get('delta_RMSE_w'):+.2f} s)"


def main() -> None:
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    from v3_protocol import shifted_no_aobt3_expr
    from v4b_features import load_v4b
    from v4b_final import combine
    from v6_prefix_gate import PrefixGate, blend_p1
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2025], help="semillas de V4b ya calculadas")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--prefix-gate", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} ya existe: no se sobrescribe (bórrala o usa --out)")
    t0 = time.time()
    need = [V4B_FINAL / f"{c}__s{s}__rank.npy" for s in args.seeds for c in ("c2_n", "c1_clf", "c0_clf")]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Faltan predicciones de V4b ({missing}); ejecuta antes src/v4b_run_candidate.sh")
    lgb_path = V5_FINAL / f"lgb_const__s{LGB_SEED}__rank.npy"
    if not lgb_path.exists():
        raise SystemExit(f"Falta {lgb_path.name}; ejecuta antes src/v5a_final.py (entrena LightGBM y lo deja en caché)")
    protected = {n: sha256(SUBMISSIONS_DIR / n) for n in PROTECTED if (SUBMISSIONS_DIR / n).exists()}
    use_pg, pg_reason = prefix_gate_enabled(args.prefix_gate)
    log(f"compuerta por prefijo: {'ACTIVADA' if use_pg else 'desactivada'} ({pg_reason})")

    rank = load_v4b("rank")
    assert rank[TARGET].null_count() == rank.height, "el target del ranking debe estar oculto"
    train = load_v4b("train")
    assert train["MVT_TIME_UTC_mvt"].dt.year().unique().to_list() == [2025]
    rule = fit_lobt_rule(train)                       # filas crudas de 2025 (sin filtro de outliers), igual que en la validación
    assert 0.9 < rule["c"] < 1.1 and rule["n"] >= 100, rule
    log(f"regla LOBT: y ≈ {rule['a']:.1f} + {rule['c']:.4f}·(MVT − LOBT), n = {rule['n']}, y > 1 h en {rule['share_y_gt_1h']:.0%}")
    pg_info, pg_mask, pg_p = {"enabled": use_pg, "reason": pg_reason}, None, None
    if use_pg:
        gate = PrefixGate().fit(train)
        pg_mask, pg_p = gate.predict(rank)
        pg_info.update({"fit_rows": gate.n, "fit_pos_rate": gate.pos_rate, "n_prefixes": len(gate.prefixes),
                        "rows_changed": int(pg_mask.sum())})
        log(f"compuerta por prefijo: {int(pg_mask.sum())} filas del ranking")
    del train

    lgb = np.load(lgb_path).astype(np.float64)
    assert lgb.shape[0] == rank.height and np.isfinite(lgb).all()
    s = rank["mvt_minus_sched"].cast(pl.Float64).to_numpy()
    in_cell = rank.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy()
    L = rank["mvt_minus_lobt"].cast(pl.Float64).to_numpy()
    v5a, v6 = [], []
    for sd in args.seeds:
        c2n = np.load(V4B_FINAL / f"c2_n__s{sd}__rank.npy").astype(np.float64)
        p1 = np.load(V4B_FINAL / f"c1_clf__s{sd}__rank.npy").astype(np.float64)
        P0 = np.load(V4B_FINAL / f"c0_clf__s{sd}__rank.npy").astype(np.float64)
        assert c2n.shape[0] == p1.shape[0] == P0.shape[0] == rank.height
        b = 0.5 * c2n + 0.5 * lgb
        v5a.append(combine(P0, b, p1, s, in_cell))
        p1v6 = blend_p1(p1, pg_mask, pg_p) if use_pg else p1
        v6.append(combine(P0, b, p1v6, s, in_cell))
    base = np.mean(v5a, axis=0)
    pred = apply_lobt_rule(np.mean(v6, axis=0), L, rule)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    rule_mask = np.isfinite(L) & (L > LOBT_GAP_SEC)
    d_rule = pred[rule_mask] - base[rule_mask]
    adep = rank["ADEP_mvt"].to_numpy()
    rule_info = {**rule, "rows_affected": int(rule_mask.sum()),
                 "by_airport": {k: int(v) for k, v in zip(*np.unique(adep[rule_mask], return_counts=True))},
                 "raised": int((d_rule > 0).sum()), "lowered": int((d_rule < 0).sum()),
                 "median_change_sec": float(np.median(d_rule)), "max_abs_change_sec": float(np.abs(d_rule).max())}

    template = read_submitting()
    sub = assemble_submission(pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred}), template)
    before = validate_submission(sub, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_parquet(out_path)
    re = pl.read_parquet(out_path); after = validate_submission(re, template); assert re.equals(sub)
    for n, h in protected.items():
        assert sha256(SUBMISSIONS_DIR / n) == h, f"{n} cambió durante la ejecución"
    diffs = {}
    for name in ("candidate_v4b.parquet", "candidate_v5a.parquet"):
        p = SUBMISSIONS_DIR / name
        if p.exists():
            j = re.join(pl.read_parquet(p), on=ID_COL, suffix="_old")
            d = (j[TARGET] - j[f"{TARGET}_old"]).abs().to_numpy()
            diffs[name] = {"rows_changed": int((d > 0).sum()), "mean_abs": float(d.mean()),
                           "p99_abs": float(np.quantile(d, 0.99)), "max_abs": float(d.max())}
    meta = {"file": str(out_path.relative_to(PROJECT_ROOT)) if out_path.is_relative_to(PROJECT_ROOT) else str(out_path),
            "sha256": sha256(out_path), "rows": re.height, "schema": {k: str(v) for k, v in re.schema.items()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "V6 = V5a (V4b C2, variante S, rama normal 0.5·CatBoost + 0.5·LightGBM) + regla de hueco LOBT"
                     + (" + compuerta por prefijo en LIRF sin NM" if use_pg else ""),
            "validation": {"lobt_rule": "outputs/experiments/v6/protocol.md §1 (−11.49 s RMSE_w ene/jul 2025, IC95 [−34.09, −0.28])",
                           "prefix_gate": "outputs/experiments/v6/prefix_gate_check.json",
                           "v5a_blend": "outputs/experiments/v5/outer_check.json (−1.41 s)"},
            "lobt_rule": rule_info, "prefix_gate": pg_info, "seeds_v4b": args.seeds, "lgb_seed": LGB_SEED,
            "lgb_cache_sha256": sha256(lgb_path), "diffs": diffs,
            "checks_before_write": before, "checks_after_reopen": after, "protected_unchanged": protected,
            "uploaded_to_bucket": False, "minutes": round((time.time() - t0) / 60, 1)}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(json.dumps({"file": meta["file"], "rows": re.height, "sha256": meta["sha256"], "ok": after["ok"],
                    "rule_rows": rule_info["rows_affected"], "prefix_gate": use_pg, "diffs": diffs}, ensure_ascii=False))
    log("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
