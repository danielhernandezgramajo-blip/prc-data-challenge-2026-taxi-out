"""
V5 (exploración) — Evaluación en los meses internos (feb + ago 2025) de ramas normales alternativas frente a la de V4b (c2_n),
usando la misma compuerta (c1_clf) y la misma celda de fecha desplazada (c0_clf) de V4b. Enero/julio no se miran.

    python src/v5_eval_inner.py --fold LOJO --cands lgb_const lgb_linear
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import MODELS_DIR  # noqa: E402
from v2_protocol import DAY_SEC  # noqa: E402

V4B = MODELS_DIR / "experiments" / "v4b" / "preds"
V5 = MODELS_DIR / "experiments" / "v5" / "preds"


def load(fold: str):
    m = dict(np.load(V4B / f"meta__{fold}__inner.npz", allow_pickle=True))
    feats = pl.read_parquet(MODELS_DIR / "experiments" / "v4b" / "features_train.parquet",
                            columns=["MVT_ID_mvt", "mvt_minus_aobt3", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt"])
    f = pl.DataFrame({"MVT_ID_mvt": m["ids"]}).join(feats, on="MVT_ID_mvt", how="left")
    assert np.array_equal(f["MVT_ID_mvt"].to_numpy(), m["ids"])
    m["grpA"] = f.select(pl.all_horizontal([pl.col(c).is_null() for c in f.columns[1:]]))[:, 0].to_numpy()
    return m


def combine(b, p1, P0, s, cell):
    out = b + p1 * (s - b)
    cm = P0[:, 0] * b + P0[:, 1] * s + P0[:, 2] * (DAY_SEC + b)
    return np.maximum(np.where(cell, cm, out), 0.0)


def rmse(e2, msk=None):
    return float(np.sqrt(e2.mean() if msk is None else e2[msk].mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="LOJO")
    ap.add_argument("--cands", nargs="+", default=["lgb_const", "lgb_linear"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate", default=None, help="p.ej. lgb_gate: añade filas con p1 = media(c1_clf, gate)")
    args = ap.parse_args()
    m = load(args.fold)
    y = m["y"].astype(np.float64); s = m["s"]; rc = m["regime_class"]; cell = m["in_cell"]; mon = m["month"]
    apt = m["airport"]; A = m["grpA"]
    sd = args.seed
    p1 = np.load(V4B / f"c1_clf__{args.fold}__s{sd}__inner.npy").astype(np.float64)
    P0 = np.load(V4B / f"c0_clf__{args.fold}__s{sd}__inner.npy").astype(np.float64)
    bs = {"c2_n": np.load(V4B / f"c2_n__{args.fold}__s{sd}__inner.npy").astype(np.float64)}
    for c in args.cands:
        p = V5 / f"{c}__{args.fold}__s{sd}__inner.npy"
        if p.exists():
            bs[c] = np.load(p).astype(np.float64)
        for it in (500, 1000, 1500):
            q = V5 / f"{c}__{args.fold}__s{sd}__inner__it{it}.npy"
            if q.exists():
                bs[f"{c}@{it}"] = np.load(q).astype(np.float64)
    for c in [k for k in bs if k != "c2_n" and "@" not in k]:
        bs[f"blend(c2_n,{c})"] = 0.5 * bs["c2_n"] + 0.5 * bs[c]
    gates = {"": p1}
    if args.gate:
        pg = np.load(V5 / f"{args.gate}__{args.fold}__s{sd}__inner.npy").astype(np.float64)
        gates[f" + gate½{args.gate}"] = 0.5 * p1 + 0.5 * pg
        from sklearn.metrics import roc_auc_score
        lab = rc == 1
        print(f"AUC clase 1: c1_clf {roc_auc_score(lab, p1):.4f} | {args.gate} {roc_auc_score(lab, pg):.4f} | media {roc_auc_score(lab, 0.5 * p1 + 0.5 * pg):.4f}")
    bulk = (rc == 0) & (y <= 3600)
    c0 = (rc == 0) & (y <= 10800)
    noLTFMA = ~(A & (apt == "LTFM"))
    rows = []
    for (k, b), (gk, pp) in [(kb, g) for kb in bs.items() for g in gates.items()]:
        e2b = (b - y) ** 2
        pr = combine(b, pp, P0, s, cell); e2 = (pr - y) ** 2
        rows.append({"normal": k + gk, "bulk_RMSE": rmse(e2b, bulk), "bulk_feb": rmse(e2b, bulk & (mon == 2)), "bulk_ago": rmse(e2b, bulk & (mon == 8)),
                     "c0<=3h_RMSE": rmse(e2b, c0), "TOTAL": rmse(e2), "TOTAL_feb": rmse(e2, mon == 2), "TOTAL_ago": rmse(e2, mon == 8),
                     "TOTAL_sin_LTFM_A": rmse(e2, noLTFMA)})
    df = pl.DataFrame(rows)
    base = df.filter(pl.col("normal") == "c2_n").row(0, named=True)
    df = df.with_columns(*(pl.col(c) - base[c] for c in ["bulk_RMSE", "TOTAL", "TOTAL_sin_LTFM_A"]) and [
        (pl.col("bulk_RMSE") - base["bulk_RMSE"]).alias("Δbulk"), (pl.col("TOTAL") - base["TOTAL"]).alias("ΔTOTAL"),
        (pl.col("TOTAL_sin_LTFM_A") - base["TOTAL_sin_LTFM_A"]).alias("ΔTOTAL_sinLTFMA")])
    with pl.Config(tbl_cols=20, tbl_width_chars=250, float_precision=1):
        print(f"fold {args.fold} seed {sd} — meses internos feb+ago (n={y.size:,}; bulk = clase 0 y ≤ 1 h, n={bulk.sum():,})")
        print(df)


if __name__ == "__main__":
    main()
