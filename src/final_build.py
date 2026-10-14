"""
FASE FINAL (14-oct-2026) — submission única de 4 meses (ene/feb/jun/jul 2026).

- Enero y julio: se reutilizan EXACTAMENTE las predicciones de candidate_v7 (mismos MVT_ID; mismo modelo V7).
- Febrero y junio (filas nuevas): modelo V7 reentrenado con los 12 meses de 2025 en la nube, con las piezas rápidas:
  rama normal LightGBM (la de V5a, sin la CatBoost por falta de tiempo), compuerta LightGBM (v5_gate), clasificador de
  celda RegimeClassifier2 (el de V4b), regla LOBT (V6) y corrección V7c. Ningún dato de 2026 entra en el ajuste.

    PRC_DATA_DIR=data/final python src/final_build.py      # data/final/ranking.parquet = filas de feb+jun del final_ranking
"""
from __future__ import annotations

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

from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, MODELS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR, TARGET  # noqa: E402

FINAL_DIR = MODELS_DIR / "experiments" / "final"
OUT = SUBMISSIONS_DIR / "candidate_final.parquet"
V7 = SUBMISSIONS_DIR / "candidate_v7.parquet"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import make_submission
    import v3_protocol
    import v4b_features
    from io_utils import read_submitting
    from make_submission import assemble_submission, validate_submission
    from modeling_utils import apply_outlier_strategy
    from v2_features import RelativeCongestion
    from v2_protocol import STRATEGY
    from v3_protocol import shifted_no_aobt3_expr
    from v4b_final import combine
    from v5_gate import LGBGate
    from v5_lgb import LGBNormal
    from v6_final import apply_lobt_rule, fit_lobt_rule
    from v7_congestion import airport_baseline, anomaly, neighbour_congestion
    from v7c_noaobt import THRESHOLD, apply_max_nb, noaobt_mask

    t0 = time.time()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("PRC_DATA_DIR", "").endswith("data/final"), "ejecuta con PRC_DATA_DIR=data/final"
    v4b_features.EXP_DIR = FINAL_DIR                     # las features de feb/jun se cachean aparte (no pisan las de ene/jul)
    train = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet")
    assert train["MVT_TIME_UTC_mvt"].dt.year().unique().to_list() == [2025]
    rank = v4b_features.load_v4b("rank")
    months = sorted(rank["MVT_TIME_UTC_mvt"].dt.month().unique().to_list())
    assert months == [2, 6], months
    assert rank[TARGET].null_count() == rank.height
    log(f"features feb/jun: {rank.height:,} salidas ({(time.time() - t0) / 60:.1f} min)")

    # --- rule + V7c inputs (sin modelos) ---
    rule = fit_lobt_rule(train)
    assert 0.9 < rule["c"] < 1.1 and rule["n"] >= 100, rule
    nb_tr = neighbour_congestion(train); base = airport_baseline(train, nb_tr); del nb_tr
    nb = neighbour_congestion(rank); nba = anomaly(rank, nb, base)
    s = rank["mvt_minus_sched"].cast(pl.Float64).to_numpy()
    in_cell = rank.select(shifted_no_aobt3_expr().fill_null(False).alias("c"))["c"].to_numpy()
    L = rank["mvt_minus_lobt"].cast(pl.Float64).to_numpy()
    mask7c = noaobt_mask(nba, rank["has_aobt3"].to_numpy(), L, in_cell, nb, THRESHOLD)
    log(f"regla LOBT n={rule['n']} | celda {int(in_cell.sum())} | V7c {int(mask7c.sum())} | L>2h {int((np.nan_to_num(L) > 7200).sum())}")

    # --- modelos (12 meses de 2025) ---
    train_o, _ = apply_outlier_strategy(train, STRATEGY)
    del train; gc.collect()
    rc = RelativeCongestion().fit(train_o)
    train_rc, rank_rc = rc.transform(train_o), rc.transform(rank)
    del train_o; gc.collect()
    info = {}

    def cached(name: str, fn):
        p = FINAL_DIR / f"{name}__febjun.npy"
        if p.exists():
            log(f"{name}: en caché"); return np.load(p).astype(np.float64)
        t1 = time.time(); out = fn(); np.save(p, out.astype(np.float32)); gc.collect()
        log(f"{name}: hecho en {(time.time() - t1) / 60:.1f} min"); return out.astype(np.float32).astype(np.float64)

    def fit_gate():
        m = LGBGate(42).fit(train_rc); info["gate"] = {k: v for k, v in m.info.items() if k != "top_gain"}; return m.predict(rank_rc)

    def fit_normal():
        m = LGBNormal(42, "const", iterations=2000).fit(train_rc); info["lgb"] = {k: v for k, v in m.info.items() if k != "top_gain"}
        return m.predict(rank_rc)

    def fit_cell():
        v3_protocol.SEED = 42
        m = v3_protocol.RegimeClassifier2(FEATURE_NUM_COLS, FEATURE_CAT_COLS).fit(train_rc)
        info["c0"] = {k: v for k, v in m.info.items() if k not in ("feature_importance", "cells_band", "nested_selection")}
        return np.asarray(m.predict_proba(rank_rc, mode="nested"), dtype=np.float64)

    p1 = cached("gate", fit_gate)
    assert p1.shape == (rank.height,) and np.isfinite(p1).all() and (p1 >= 0).all() and (p1 <= 1).all()
    b = cached("lgb_const", fit_normal)
    assert b.shape == (rank.height,) and np.isfinite(b).all()
    try:
        if os.environ.get("SKIP_C0"):
            raise RuntimeError("SKIP_C0")
        P0 = cached("c0_clf", fit_cell)
        assert P0.shape == (rank.height, 3) and np.isfinite(P0).all()
    except Exception as e:                                   # sin clasificador de celda: la celda usa solo la compuerta
        log(f"c0 falló ({e!r}); celda con compuerta solamente")
        P0 = np.column_stack([1 - p1, p1, np.zeros_like(p1)])
    del train_rc, rank_rc; gc.collect()

    # --- V7 ---
    pred = combine(P0, apply_max_nb(b, nb, mask7c), p1, s, in_cell)
    pred = apply_lobt_rule(pred, L, rule)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    new = pl.DataFrame({ID_COL: rank[ID_COL], "pred": pred})
    log(f"feb/jun: media {pred.mean():.0f} s, p99 {np.quantile(pred, 0.99):.0f}, máx {pred.max():.0f}")

    # --- ene/jul desde V7 + ensamblado con la plantilla final ---
    v7 = pl.read_parquet(V7).rename({TARGET: "pred"}).with_columns(pl.col("pred").cast(pl.Float64))
    allp = pl.concat([v7, new])
    assert allp[ID_COL].n_unique() == allp.height
    template = read_submitting()
    make_submission.EXPECTED_SUBMITTING_ROWS = template.height
    assert set(allp[ID_COL].to_list()) == set(template[ID_COL].to_list()), "los IDs no coinciden con final_submitting"
    sub = assemble_submission(allp, template)
    before = validate_submission(sub, template)
    if OUT.exists():
        raise FileExistsError(OUT)
    sub.write_parquet(OUT)
    re = pl.read_parquet(OUT); after = validate_submission(re, template); assert re.equals(sub)
    j = re.join(pl.read_parquet(V7), on=ID_COL, suffix="_v7", how="inner")
    assert j.height == v7.height and (j[TARGET] == j[f"{TARGET}_v7"]).all(), "ene/jul deben ser idénticos a V7"
    mon = re.join(rank.select(ID_COL, pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m")), on=ID_COL, how="left")
    by_month = {str(k): int(v) for k, v in mon.group_by("m").len().sort("m").iter_rows()}
    meta = {"file": str(OUT.relative_to(PROJECT_ROOT)), "sha256": sha256(OUT), "rows": re.height,
            "schema": {k: str(v) for k, v in re.schema.items()}, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "FINAL: ene/jul = candidate_v7 (idénticas); feb/jun = V7 reentrenado (LightGBM normal + compuerta LightGBM + "
                     "RegimeClassifier2 + regla LOBT + V7c), 12 meses de 2025, semilla 42",
            "rows_by_month_febjun_null_means_janjul": by_month, "lobt_rule": rule, "v7c_rows": int(mask7c.sum()),
            "cell_rows": int(in_cell.sum()), "models": info, "v7_sha256": sha256(V7),
            "checks_before_write": before, "checks_after_reopen": after, "uploaded_to_bucket": False,
            "minutes": round((time.time() - t0) / 60, 1)}
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(json.dumps({"file": meta["file"], "sha256": meta["sha256"], "rows": re.height, "ok": after["ok"]}, ensure_ascii=False))
    log("NO se ha subido nada al bucket del concurso.")


if __name__ == "__main__":
    main()
