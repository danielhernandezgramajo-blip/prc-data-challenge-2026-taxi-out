"""
V2 — Evaluación de candidatos sobre las predicciones de validación guardadas (sin reentrenar nada).

Candidatos (registrados en experiment_registry.csv ANTES de mirar resultados; ver REGISTRY):
    V1P            0.5·direct + 0.5·residual (v1_protocol)  ← V1 bajo el protocolo de 8 folds (baseline congelada)
    V1P_direct / V1P_residual                                 componentes (información)
    S_mix          p0·normal + p1·(MVT−SCHED) + p2·(86 400 + normal), normal = 0.5·direct_normal + 0.5·residual
    S_mix_dn       igual con normal = direct_normal
    S_mix_onV1P    igual con normal = V1P (regresor "contaminado" por los extremos; control)
    S_mix_floor    S_mix con suelo pred ≥ (MVT − AOBT_3) − 3 600 s cuando existe AOBT_3

Métricas por fold, pooled (todas las filas), por aeropuerto, cobertura ±5/10/15 min, bootstrap pareado por
bloques aeropuerto-día (1 000 remuestreos, SEED 42), intervalos conformales de diagnóstico y criterios de
selección. Diciembre NO se lee aquí salvo con --holdout, que solo debe usarse una vez con el finalista.

Uso: python src/v2_analysis.py              # folds de desarrollo
     python src/v2_analysis.py --holdout    # añade diciembre (una sola vez)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import OUTLIER_HI_SEC, SEED  # noqa: E402
from v2_metrics import COVER_S, block, by_airport, rmse  # noqa: E402
from v2_protocol import DAY_SEC, DEV_FOLDS, HOLDOUT_FOLD, OUT_DIR, PRED_DIR  # noqa: E402

REGISTRY = [
    {"candidate": "V1P", "family": "baseline", "definition": "0.5*direct + 0.5*residual (v1_protocol, features V1, 1000 it., drop_nonpositive)", "fits_reused": "v1_protocol"},
    {"candidate": "V1P_direct", "family": "baseline-component", "definition": "catboost directo v1_protocol", "fits_reused": "v1_protocol"},
    {"candidate": "V1P_residual", "family": "baseline-component", "definition": "catboost residual v1_protocol (fallback mediana jerárquica sin AOBT_3)", "fits_reused": "v1_protocol"},
    {"candidate": "S_mix", "family": "S (mezcla por régimen)", "definition": "p0*normal + p1*(MVT-SCHED) + p2*(86400+normal); normal = 0.5*direct_normal + 0.5*residual", "fits_reused": "S_mixture + v1_protocol.residual"},
    {"candidate": "S_mix_dn", "family": "S (mezcla por régimen)", "definition": "como S_mix con normal = direct_normal", "fits_reused": "S_mixture"},
    {"candidate": "S_mix_onV1P", "family": "S (control)", "definition": "como S_mix con normal = V1P (regresor no depurado de extremos)", "fits_reused": "S_mixture.proba + v1_protocol"},
    {"candidate": "S_mix_floor", "family": "S + post-proceso", "definition": "S_mix con suelo pred >= (MVT-AOBT_3) - 3600 s si hay AOBT_3", "fits_reused": "S_mixture + v1_protocol.residual"},
]
BASELINE = "V1P"
CRITERIA = {"min_rmse_gain_s": 3.0, "min_cov5_gain_pp": 0.5, "min_folds_not_worse": 5, "max_fold_worsening_pct": 2.0,
            "min_cov5_pooled": 87.0, "min_cov5_each_fold": 85.0, "min_p_improve": 0.87, "n_boot": 1000}


def load_fold(fold: str) -> dict:
    meta = np.load(PRED_DIR / f"meta__{fold}.npz", allow_pickle=False)
    d = {k: meta[k] for k in meta.files}
    d["y"] = d["y"].astype(np.float64)
    for name in ["v1_protocol__direct", "v1_protocol__residual", "S_mixture__direct_normal"]:
        var, kind = name.split("__")
        f = PRED_DIR / f"{var}__{fold}__{kind}.npy"
        d[name] = np.load(f).astype(np.float64) if f.exists() else None
    f = PRED_DIR / f"S_mixture__{fold}__proba.npy"
    d["proba"] = np.load(f).astype(np.float64) if f.exists() else None
    return d


def candidates(d: dict) -> dict[str, np.ndarray]:
    out = {}
    pd_, pr_ = d["v1_protocol__direct"], d["v1_protocol__residual"]
    if pd_ is None or pr_ is None:
        return out
    v1p = 0.5 * pd_ + 0.5 * pr_
    out["V1P"] = v1p; out["V1P_direct"] = pd_; out["V1P_residual"] = pr_
    if d["proba"] is not None and d["S_mixture__direct_normal"] is not None:
        p = d["proba"]; sched = np.nan_to_num(d["mvt_minus_sched"].astype(np.float64), nan=0.0)
        dn = d["S_mixture__direct_normal"]

        def mix(normal):
            return p[:, 0] * normal + p[:, 1] * sched + p[:, 2] * (DAY_SEC + normal)
        out["S_mix"] = mix(0.5 * dn + 0.5 * pr_)
        out["S_mix_dn"] = mix(dn)
        out["S_mix_onV1P"] = mix(v1p)
        base = d["mvt_minus_aobt3"].astype(np.float64)
        floor = np.where(np.isnan(base), -np.inf, base - 3600.0)
        out["S_mix_floor"] = np.maximum(out["S_mix"], floor)
    return out


def paired_block_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, blocks: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap pareado por bloques (aeropuerto-día): remuestrea bloques con reemplazo y recalcula RMSE de A y B
    sobre exactamente las mismas filas. Devuelve la distribución de ΔRMSE = RMSE_B − RMSE_A."""
    ea = (pa - y) ** 2; eb = (pb - y) ** 2
    _, inv = np.unique(blocks, return_inverse=True)
    nb = inv.max() + 1
    sa = np.bincount(inv, weights=ea, minlength=nb); sb = np.bincount(inv, weights=eb, minlength=nb); cnt = np.bincount(inv, minlength=nb).astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, nb)
        n = cnt[idx].sum()
        deltas[i] = np.sqrt(sb[idx].sum() / n) - np.sqrt(sa[idx].sum() / n)
    return {"delta_mean": float(deltas.mean()), "delta_p2.5": float(np.quantile(deltas, 0.025)), "delta_p50": float(np.median(deltas)),
            "delta_p97.5": float(np.quantile(deltas, 0.975)), "p_improve": float((deltas < 0).mean()), "n_blocks": int(nb)}


def conformal(folds: list[str], data: dict, preds: dict, cand: str, alphas=(0.13, 0.10)) -> list[dict]:
    """Intervalo |y − pred| ≤ q, con q = cuantil (1−α) de los residuos absolutos de TODOS los folds anteriores;
    se evalúa en el fold siguiente (cobertura real, ancho medio = 2q). Diagnóstico; no va a la submission."""
    rows = []
    for i in range(1, len(folds)):
        cal = np.concatenate([np.abs(preds[f][cand] - data[f]["y"]) for f in folds[:i]])
        f = folds[i]; ae = np.abs(preds[f][cand] - data[f]["y"])
        for a in alphas:
            q = float(np.quantile(cal, 1 - a))
            rows.append({"candidate": cand, "fold": f, "nominal": 1 - a, "q_half_width_s": q, "coverage": float((ae <= q).mean()),
                         "n_cal": int(cal.size), "n_eval": int(ae.size)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="incluye diciembre (SOLO una vez, con el finalista ya elegido)")
    args = ap.parse_args()
    folds = DEV_FOLDS + ([HOLDOUT_FOLD] if args.holdout else [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(REGISTRY).write_csv(OUT_DIR / "experiment_registry.csv")

    data = {f: load_fold(f) for f in folds}
    preds = {f: candidates(data[f]) for f in folds}
    cands = [c for c in REGISTRY_NAMES() if all(c in preds[f] for f in DEV_FOLDS)]
    assert BASELINE in cands, "faltan componentes de v1_protocol"

    # --- métricas por fold y pooled -------------------------------------------------------------
    rows, pooled_rows, ap_rows, cov_rows = [], [], [], []
    for c in cands:
        for f in folds:
            y, p = data[f]["y"], preds[f][c]
            rows.append({"candidate": c, "fold": f, "set": "holdout" if f == HOLDOUT_FOLD else "dev", **block(y, p)})
            ap_ = by_airport(data[f]["airport"], y, p).with_columns(pl.lit(c).alias("candidate"), pl.lit(f).alias("fold"))
            ap_rows.append(ap_)
        yd = np.concatenate([data[f]["y"] for f in DEV_FOLDS]); pd_ = np.concatenate([preds[f][c] for f in DEV_FOLDS])
        pooled_rows.append({"candidate": c, "set": "dev_pooled", **block(yd, pd_),
                            "rmse_unweighted_fold_mean": float(np.mean([rmse(data[f]["y"], preds[f][c]) for f in DEV_FOLDS]))})
        if args.holdout:
            pooled_rows.append({"candidate": c, "set": "holdout_Dec", **block(data[HOLDOUT_FOLD]["y"], preds[HOLDOUT_FOLD][c]), "rmse_unweighted_fold_mean": np.nan})
        apd = by_airport(np.concatenate([data[f]["airport"] for f in DEV_FOLDS]), yd, pd_).with_columns(pl.lit(c).alias("candidate"), pl.lit("dev_pooled").alias("fold"))
        ap_rows.append(apd)
    metrics = pl.from_dicts(rows, infer_schema_length=None); metrics.write_csv(OUT_DIR / "metrics_by_fold.csv")
    pooled = pl.from_dicts(pooled_rows, infer_schema_length=None); pooled.write_csv(OUT_DIR / "metrics_pooled.csv")
    pl.concat(ap_rows, how="vertical_relaxed").write_csv(OUT_DIR / "metrics_by_airport.csv")
    cov = metrics.select(["candidate", "fold", "set", "n"] + [f"cov_le{s}" for s in COVER_S] + ["ae_p50", "ae_p87", "ae_p90", "ae_p95", "ae_p99", "ae_max"])
    cov.write_csv(OUT_DIR / "coverage_by_fold.csv")

    # --- bootstrap pareado por bloques aeropuerto-día (sobre las filas pooled de desarrollo) --------------
    yd = np.concatenate([data[f]["y"] for f in DEV_FOLDS])
    blocks = np.concatenate([np.char.add(np.char.add(data[f]["airport"].astype(str), "_"), data[f]["day"].astype(str)) for f in DEV_FOLDS])
    base_p = np.concatenate([preds[f][BASELINE] for f in DEV_FOLDS])
    boot_rows = []
    for c in cands:
        if c == BASELINE:
            continue
        pc = np.concatenate([preds[f][c] for f in DEV_FOLDS])
        b = paired_block_bootstrap(yd, base_p, pc, blocks, CRITERIA["n_boot"], SEED)
        boot_rows.append({"candidate": c, "vs": BASELINE, "rmse_baseline": rmse(yd, base_p), "rmse_candidate": rmse(yd, pc),
                          "delta_rmse_point": rmse(yd, pc) - rmse(yd, base_p), **b})
    boot = pl.DataFrame(boot_rows); boot.write_csv(OUT_DIR / "bootstrap_comparison.csv")

    # --- criterios de selección (solo desarrollo) -----------------------------------------------------
    sel_rows = []
    base_fold = {f: rmse(data[f]["y"], preds[f][BASELINE]) for f in DEV_FOLDS}
    base_pool = pooled.filter((pl.col("candidate") == BASELINE) & (pl.col("set") == "dev_pooled")).row(0, named=True)
    for c in cands:
        if c == BASELINE:
            continue
        cp = pooled.filter((pl.col("candidate") == c) & (pl.col("set") == "dev_pooled")).row(0, named=True)
        fr = {f: rmse(data[f]["y"], preds[f][c]) for f in DEV_FOLDS}
        not_worse = sum(fr[f] <= base_fold[f] for f in DEV_FOLDS)
        worst_pct = max((fr[f] - base_fold[f]) / base_fold[f] * 100 for f in DEV_FOLDS)
        cov_each = min(metrics.filter((pl.col("candidate") == c) & (pl.col("set") == "dev"))["cov_le300"].to_list())
        bb = boot.filter(pl.col("candidate") == c).row(0, named=True)
        gain = base_pool["rmse"] - cp["rmse"]; cov_gain = cp["cov_le300"] - base_pool["cov_le300"]
        # dependencia de pocas filas: ganancia de SSE aportada por las 50 filas con mayor reducción de error²
        pc = np.concatenate([preds[f][c] for f in DEV_FOLDS]); e_b = (base_p - yd) ** 2; e_c = (pc - yd) ** 2
        gains = e_b - e_c; top50 = np.sort(gains)[-50:].sum(); total_gain = gains.sum()
        share_top50 = float(top50 / total_gain) if total_gain > 0 else np.nan
        normal = (yd > 0) & (yd <= OUTLIER_HI_SEC)
        crit = {
            "c1_rmse_pooled_lower": cp["rmse"] < base_pool["rmse"],
            "c2_practical_gain": (gain >= CRITERIA["min_rmse_gain_s"] and cov_gain > -0.5) or (cov_gain >= CRITERIA["min_cov5_gain_pp"] and gain > -3.0),
            "c3_folds_not_worse": not_worse >= CRITERIA["min_folds_not_worse"],
            "c4_no_fold_worse_2pct": worst_pct <= CRITERIA["max_fold_worsening_pct"],
            "c5_cov5_pooled_ge_87": cp["cov_le300"] >= CRITERIA["min_cov5_pooled"],
            "c6_cov5_each_fold_ge_85": cov_each >= CRITERIA["min_cov5_each_fold"],
            "c8_not_few_rows": share_top50 < 0.9 if not np.isnan(share_top50) else True,
            "boot_p_improve_ge_0.87": bb["p_improve"] >= CRITERIA["min_p_improve"],
        }
        sel_rows.append({"candidate": c, "rmse_pooled": cp["rmse"], "rmse_gain_s": gain, "cov5_pooled": cp["cov_le300"], "cov5_gain_pp": cov_gain,
                         "folds_not_worse": f"{not_worse}/7", "worst_fold_worsening_pct": worst_pct, "min_cov5_fold": cov_each,
                         "rmse_normal_gain_s": float(np.sqrt(np.mean(e_b[normal])) - np.sqrt(np.mean(e_c[normal]))),
                         "share_gain_top50_rows": share_top50, "p_improve_boot": bb["p_improve"], **crit,
                         "all_criteria": all(crit.values())})
    sel = pl.from_dicts(sel_rows, infer_schema_length=None); sel.write_csv(OUT_DIR / "selection_criteria.csv")

    # --- conformal (diagnóstico) sobre desarrollo -----------------------------------------------------
    conf = []
    for c in [BASELINE] + [c for c in cands if c.startswith("S_mix")]:
        conf += conformal(DEV_FOLDS, data, preds, c)
    pl.DataFrame(conf).write_csv(OUT_DIR / "conformal_intervals.csv")

    (OUT_DIR / "analysis_run.json").write_text(json.dumps({"folds": folds, "holdout_opened": args.holdout, "candidates": cands,
                                                          "criteria": CRITERIA, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(50)
    print(pooled.select("candidate", "set", "rmse", "rmse_unweighted_fold_mean", "mae", "medae", "cov_le300", "cov_le600", "cov_le900", "rmse_normal", "rmse_extreme", "n_pred_negative"))
    print(metrics.filter(pl.col("candidate").is_in([BASELINE, "S_mix"])).select("candidate", "fold", "rmse", "mae", "cov_le300", "n_extreme", "rmse_extreme"))
    print(boot.select("candidate", "rmse_baseline", "rmse_candidate", "delta_rmse_point", "delta_p2.5", "delta_p97.5", "p_improve"))
    print(sel)


def REGISTRY_NAMES() -> list[str]:
    return [r["candidate"] for r in REGISTRY]


if __name__ == "__main__":
    main()
