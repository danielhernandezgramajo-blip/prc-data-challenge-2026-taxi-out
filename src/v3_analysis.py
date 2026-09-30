"""
V3 (etapa 2 de V2) — Evaluación de candidatos sobre predicciones de validación guardadas (V2 + V3), sin reentrenar.

Candidatos (registrados en experiment_registry.csv ANTES de mirar resultados; ver REGISTRY):
    V1P, S_mix          referencias (componentes de V2; S_mix = candidate_v2)
    S2_mix_a0           proba2 (features V1 + days_after_sched, SIN celdas) + normal_v2          ablación
    S2_mix_a5 / _a100   proba2 con α = 5 / 100 + normal_v2                                         sensibilidad
    S2_mix              proba2 con α = 20 (PRIMARIO, prefijado) + normal_v2                        elegible
    A_mix               proba V2 + normal_s = 0.5·direct_normal_s + 0.5·residual_normal_s          elegible
    A_mix_dn / A_mix_rn proba V2 + un solo componente de superficie                                información
    A_mix_4             proba V2 + 0.25·(direct_normal + residual_V1 + direct_normal_s + residual_normal_s)   elegible
    S2A_mix             proba2 (α = 20) + normal_s                                                 elegible
    S2A_mix_4           proba2 (α = 20) + normal 4 componentes                                     elegible
    normal_v2 = 0.5·direct_normal + 0.5·residual (v1_protocol), es decir, la rama normal de S_mix.

Regla de decisión (prefijada antes de ver resultados): entre las ELEGIBLES, la finalista es la de menor RMSE pooled de
desarrollo que cumpla TODOS los criterios frente a S_mix y frente a V1P (c1–c6 y bootstrap P ≥ 0.87; c8 informativo,
ver informe V2). Si ninguna cumple, se mantiene S_mix (candidate_v2) y no se genera candidate_v3. α no se selecciona:
20 es el valor primario; 0/5/100 solo se reportan.

Uso: python src/v3_analysis.py                  # desarrollo
     python src/v3_analysis.py --freeze         # además escribe finalist_frozen.json (una vez)
     python src/v3_analysis.py --holdout        # añade diciembre (segunda apertura; solo con la finalista congelada)
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
from v2_analysis import CRITERIA, conformal, paired_block_bootstrap  # noqa: E402
from v2_analysis import load_fold as load_fold_v2  # noqa: E402
from v2_metrics import COVER_S, block, by_airport, rmse  # noqa: E402
from v2_protocol import DEV_FOLDS, HOLDOUT_FOLD  # noqa: E402
from v3_protocol import ALPHA_PRIMARY, ALPHAS, OUT_DIR, PRED_DIR, mixture  # noqa: E402

REGISTRY = [
    {"candidate": "V1P", "family": "referencia", "definition": "0.5*direct + 0.5*residual (v1_protocol)", "eligible": False},
    {"candidate": "S_mix", "family": "referencia (candidate_v2)", "definition": "proba V2; normal_v2 = 0.5*direct_normal + 0.5*residual_V1", "eligible": False},
    {"candidate": "S2_mix_a0", "family": "S2 (ablación)", "definition": "proba2 alpha=0 (feature days_after_sched, sin celdas) + normal_v2", "eligible": False},
    {"candidate": "S2_mix_a5", "family": "S2 (sensibilidad)", "definition": "proba2 alpha=5 + normal_v2", "eligible": False},
    {"candidate": "S2_mix", "family": "S2 (primario)", "definition": "proba2 alpha=20 + normal_v2", "eligible": True},
    {"candidate": "S2_mix_a100", "family": "S2 (sensibilidad)", "definition": "proba2 alpha=100 + normal_v2", "eligible": False},
    {"candidate": "A_mix", "family": "A (superficie)", "definition": "proba V2 + normal_s = 0.5*direct_normal_s + 0.5*residual_normal_s", "eligible": True},
    {"candidate": "A_mix_dn", "family": "A (información)", "definition": "proba V2 + direct_normal_s", "eligible": False},
    {"candidate": "A_mix_rn", "family": "A (información)", "definition": "proba V2 + residual_normal_s", "eligible": False},
    {"candidate": "A_mix_4", "family": "A (superficie)", "definition": "proba V2 + 0.25*(direct_normal + residual_V1 + direct_normal_s + residual_normal_s)", "eligible": True},
    {"candidate": "S2A_mix", "family": "S2+A", "definition": "proba2 alpha=20 + normal_s", "eligible": True},
    {"candidate": "S2A_mix_4", "family": "S2+A", "definition": "proba2 alpha=20 + normal 4 componentes", "eligible": True},
    {"candidate": "S3_mix", "family": "S3 (2c, post-hoc)", "definition": "proba2 celdas jerárquicas por banda de retraso (>=12 h), alpha=20 + normal_v2", "eligible": False},
    {"candidate": "S3A_mix", "family": "S3+A (2c, post-hoc)", "definition": "proba2 celdas jerárquicas por banda (>=12 h), alpha=20 + normal_s", "eligible": False},
    {"candidate": "S3A_mix_4", "family": "S3+A (2c, post-hoc)", "definition": "proba2 celdas jerárquicas por banda (>=12 h), alpha=20 + normal 4 componentes", "eligible": False},
]
POST_HOC = ["S3_mix", "S3A_mix", "S3A_mix_4"]   # definidos DESPUÉS de abrir diciembre: no elegibles por la regla prefijada; ver --final-choice
ELIGIBLE = [r["candidate"] for r in REGISTRY if r["eligible"]]
BASELINES = ["S_mix", "V1P"]


def load_fold(fold: str) -> dict:
    d = load_fold_v2(fold)
    for a in ALPHAS:
        f = PRED_DIR / f"S2_mixture__{fold}__proba_a{a}.npy"
        d[f"proba2_a{a}"] = np.load(f).astype(np.float64) if f.exists() else None
    f = PRED_DIR / f"S2_mixture__{fold}__proba_b{ALPHA_PRIMARY}.npy"
    d["proba2_band"] = np.load(f).astype(np.float64) if f.exists() else None
    for kind in ("direct_normal", "residual_normal"):
        f = PRED_DIR / f"A_surface__{fold}__{kind}.npy"
        d[f"A_surface__{kind}"] = np.load(f).astype(np.float64) if f.exists() else None
    return d


def candidates(d: dict) -> dict[str, np.ndarray]:
    out = {}
    pd_, pr_, dn, p = d["v1_protocol__direct"], d["v1_protocol__residual"], d["S_mixture__direct_normal"], d["proba"]
    if pd_ is None or pr_ is None:
        return out
    out["V1P"] = 0.5 * pd_ + 0.5 * pr_
    if dn is None or p is None:
        return out
    s = d["mvt_minus_sched"]
    normal_v2 = 0.5 * dn + 0.5 * pr_
    out["S_mix"] = mixture(p, normal_v2, s)
    for a in ALPHAS:
        p2 = d[f"proba2_a{a}"]
        if p2 is not None:
            out["S2_mix" if a == ALPHA_PRIMARY else f"S2_mix_a{a}"] = mixture(p2, normal_v2, s)
    dn_s, rn_s = d["A_surface__direct_normal"], d["A_surface__residual_normal"]
    if dn_s is not None and rn_s is not None:
        normal_s = 0.5 * dn_s + 0.5 * rn_s
        normal_4 = 0.25 * (dn + pr_ + dn_s + rn_s)
        out["A_mix"] = mixture(p, normal_s, s); out["A_mix_dn"] = mixture(p, dn_s, s); out["A_mix_rn"] = mixture(p, rn_s, s)
        out["A_mix_4"] = mixture(p, normal_4, s)
        p2 = d[f"proba2_a{ALPHA_PRIMARY}"]
        if p2 is not None:
            out["S2A_mix"] = mixture(p2, normal_s, s); out["S2A_mix_4"] = mixture(p2, normal_4, s)
        if d["proba2_band"] is not None:
            out["S3A_mix"] = mixture(d["proba2_band"], normal_s, s); out["S3A_mix_4"] = mixture(d["proba2_band"], normal_4, s)
    if d["proba2_band"] is not None:
        out["S3_mix"] = mixture(d["proba2_band"], normal_v2, s)
    return out


def error_decomposition(y: np.ndarray, pred: np.ndarray, cls: np.ndarray) -> dict:
    sse = (pred - y) ** 2; tot = float(sse.sum())
    out = {}
    for k in (0, 1, 2):
        m = cls == k
        out[f"n_class{k}"] = int(m.sum()); out[f"rmse_class{k}"] = float(np.sqrt(sse[m].mean())) if m.any() else np.nan
        out[f"sse_share_class{k}"] = float(sse[m].sum() / tot * 100) if tot > 0 else np.nan
    return out


def criteria_rows(cands, baseline, data, preds, metrics, pooled, boot) -> list[dict]:
    yd = np.concatenate([data[f]["y"] for f in DEV_FOLDS]); base_p = np.concatenate([preds[f][baseline] for f in DEV_FOLDS])
    base_fold = {f: rmse(data[f]["y"], preds[f][baseline]) for f in DEV_FOLDS}
    base_pool = pooled.filter((pl.col("candidate") == baseline) & (pl.col("set") == "dev_pooled")).row(0, named=True)
    normal = (yd > 0) & (yd <= OUTLIER_HI_SEC)
    rows = []
    for c in cands:
        if c == baseline:
            continue
        cp = pooled.filter((pl.col("candidate") == c) & (pl.col("set") == "dev_pooled")).row(0, named=True)
        fr = {f: rmse(data[f]["y"], preds[f][c]) for f in DEV_FOLDS}
        not_worse = sum(fr[f] <= base_fold[f] for f in DEV_FOLDS)
        worst_pct = max((fr[f] - base_fold[f]) / base_fold[f] * 100 for f in DEV_FOLDS)
        cov_each = min(metrics.filter((pl.col("candidate") == c) & (pl.col("set") == "dev"))["cov_le300"].to_list())
        bb = boot.filter((pl.col("candidate") == c) & (pl.col("vs") == baseline)).row(0, named=True)
        gain = base_pool["rmse"] - cp["rmse"]; cov_gain = cp["cov_le300"] - base_pool["cov_le300"]
        pc = np.concatenate([preds[f][c] for f in DEV_FOLDS]); e_b = (base_p - yd) ** 2; e_c = (pc - yd) ** 2
        gains = e_b - e_c; top50 = np.sort(gains)[-50:].sum(); total_gain = gains.sum()
        share_top50 = float(top50 / total_gain) if total_gain > 0 else np.nan
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
        required = {k: v for k, v in crit.items() if k != "c8_not_few_rows"}
        rows.append({"candidate": c, "vs": baseline, "rmse_pooled": cp["rmse"], "rmse_gain_s": gain, "cov5_pooled": cp["cov_le300"], "cov5_gain_pp": cov_gain,
                     "folds_not_worse": f"{not_worse}/7", "worst_fold_worsening_pct": worst_pct, "min_cov5_fold": cov_each,
                     "rmse_normal_gain_s": float(np.sqrt(np.mean(e_b[normal])) - np.sqrt(np.mean(e_c[normal]))),
                     "share_gain_top50_rows": share_top50, "p_improve_boot": bb["p_improve"], **crit,
                     "all_required_criteria": all(required.values()), "all_criteria_incl_c8": all(crit.values())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="incluye diciembre (segunda apertura; solo con la finalista congelada)")
    ap.add_argument("--freeze", action="store_true", help="escribe finalist_frozen.json según la regla prefijada (una sola vez)")
    ap.add_argument("--final-choice", default=None, help="candidata que irá a candidate_v3 si difiere de la finalista congelada (post-hoc, documentado)")
    ap.add_argument("--reason", default=None, help="motivo de la elección post-hoc (obligatorio con --final-choice)")
    args = ap.parse_args()
    folds = DEV_FOLDS + ([HOLDOUT_FOLD] if args.holdout else [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(REGISTRY).write_csv(OUT_DIR / "experiment_registry.csv")

    data = {f: load_fold(f) for f in folds}
    preds = {f: candidates(data[f]) for f in folds}
    names = [r["candidate"] for r in REGISTRY]
    cands = [c for c in names if all(c in preds[f] for f in DEV_FOLDS)]
    assert all(b in cands for b in BASELINES), "faltan componentes de V2 (v1_protocol / S_mixture)"

    rows, pooled_rows, ap_rows, dec_rows = [], [], [], []
    for c in cands:
        for f in folds:
            if c not in preds[f]:
                continue
            y, p = data[f]["y"], preds[f][c]
            rows.append({"candidate": c, "fold": f, "set": "holdout" if f == HOLDOUT_FOLD else "dev", **block(y, p),
                         **error_decomposition(y, p, data[f]["regime_class"])})
            ap_rows.append(by_airport(data[f]["airport"], y, p).with_columns(pl.lit(c).alias("candidate"), pl.lit(f).alias("fold")))
        yd = np.concatenate([data[f]["y"] for f in DEV_FOLDS]); pd_ = np.concatenate([preds[f][c] for f in DEV_FOLDS])
        cls = np.concatenate([data[f]["regime_class"] for f in DEV_FOLDS])
        pooled_rows.append({"candidate": c, "set": "dev_pooled", **block(yd, pd_), **error_decomposition(yd, pd_, cls),
                            "rmse_unweighted_fold_mean": float(np.mean([rmse(data[f]["y"], preds[f][c]) for f in DEV_FOLDS]))})
        if args.holdout and c in preds[HOLDOUT_FOLD]:
            yh, ph = data[HOLDOUT_FOLD]["y"], preds[HOLDOUT_FOLD][c]
            pooled_rows.append({"candidate": c, "set": "holdout_Dec", **block(yh, ph), **error_decomposition(yh, ph, data[HOLDOUT_FOLD]["regime_class"]),
                                "rmse_unweighted_fold_mean": np.nan})
        ap_rows.append(by_airport(np.concatenate([data[f]["airport"] for f in DEV_FOLDS]), yd, pd_).with_columns(pl.lit(c).alias("candidate"), pl.lit("dev_pooled").alias("fold")))
    metrics = pl.from_dicts(rows, infer_schema_length=None); metrics.write_csv(OUT_DIR / "metrics_by_fold.csv")
    pooled = pl.from_dicts(pooled_rows, infer_schema_length=None); pooled.write_csv(OUT_DIR / "metrics_pooled.csv")
    pl.concat(ap_rows, how="vertical_relaxed").write_csv(OUT_DIR / "metrics_by_airport.csv")
    metrics.select(["candidate", "fold", "set", "n"] + [f"cov_le{s}" for s in COVER_S] + ["ae_p50", "ae_p87", "ae_p90", "ae_p95", "ae_p99", "ae_max"]).write_csv(OUT_DIR / "coverage_by_fold.csv")

    yd = np.concatenate([data[f]["y"] for f in DEV_FOLDS])
    blocks = np.concatenate([np.char.add(np.char.add(data[f]["airport"].astype(str), "_"), data[f]["day"].astype(str)) for f in DEV_FOLDS])
    boot_rows = []
    for b in BASELINES:
        base_p = np.concatenate([preds[f][b] for f in DEV_FOLDS])
        for c in cands:
            if c == b:
                continue
            pc = np.concatenate([preds[f][c] for f in DEV_FOLDS])
            r = paired_block_bootstrap(yd, base_p, pc, blocks, CRITERIA["n_boot"], SEED)
            boot_rows.append({"candidate": c, "vs": b, "rmse_baseline": rmse(yd, base_p), "rmse_candidate": rmse(yd, pc), "delta_rmse_point": rmse(yd, pc) - rmse(yd, base_p), **r})
    boot = pl.DataFrame(boot_rows); boot.write_csv(OUT_DIR / "bootstrap_comparison.csv")

    sel_rows = []
    for b in BASELINES:
        sel_rows += criteria_rows(cands, b, data, preds, metrics, pooled, boot)
    sel = pl.from_dicts(sel_rows, infer_schema_length=None); sel.write_csv(OUT_DIR / "selection_criteria.csv")

    conf = []
    for c in ["V1P", "S_mix"] + [c for c in ELIGIBLE + POST_HOC if c in cands]:
        conf += conformal(DEV_FOLDS, data, preds, c)
    pl.DataFrame(conf).write_csv(OUT_DIR / "conformal_intervals.csv")

    # --- regla de decisión prefijada -------------------------------------------------------------------
    passing = []
    for c in ELIGIBLE:
        if c not in cands:
            continue
        ok = all(sel.filter((pl.col("candidate") == c) & (pl.col("vs") == b))["all_required_criteria"][0] for b in BASELINES)
        if ok:
            passing.append((float(pooled.filter((pl.col("candidate") == c) & (pl.col("set") == "dev_pooled"))["rmse"][0]), c))
    finalist = min(passing)[1] if passing else None
    decision = {"rule": "menor RMSE pooled de desarrollo entre las elegibles que cumplen todos los criterios requeridos frente a S_mix y V1P",
                "eligible": [c for c in ELIGIBLE if c in cands], "passing": sorted(passing), "finalist": finalist,
                "keep_candidate_v2_if_none": finalist is None, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT_DIR / "decision_dev.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.freeze:
        fz = OUT_DIR / "finalist_frozen.json"
        payload = {"finalist": finalist, "definition": next((r["definition"] for r in REGISTRY if r["candidate"] == finalist), None),
                   "alpha": ALPHA_PRIMARY, "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision_dev": decision}
        if fz.exists():
            prev = json.loads(fz.read_text(encoding="utf-8"))
            assert prev["finalist"] == finalist, f"finalist_frozen.json ya existe con otra finalista ({prev['finalist']}); no se sobrescribe"
            print("finalist_frozen.json ya existía con la misma finalista; no se modifica")
        else:
            fz.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); print("finalista congelada:", finalist)
    if args.final_choice:
        assert args.reason, "--final-choice requiere --reason"
        c = args.final_choice
        assert c in cands, f"{c} no está evaluada"
        ok = all(sel.filter((pl.col("candidate") == c) & (pl.col("vs") == b))["all_required_criteria"][0] for b in BASELINES)
        assert ok, f"{c} no cumple todos los criterios requeridos en desarrollo frente a {BASELINES}; no se puede elegir"
        fz = json.loads((OUT_DIR / "finalist_frozen.json").read_text(encoding="utf-8")) if (OUT_DIR / "finalist_frozen.json").exists() else None
        (OUT_DIR / "final_choice.json").write_text(json.dumps({
            "candidate": c, "post_hoc": c not in ELIGIBLE, "frozen_finalist": fz["finalist"] if fz else None, "reason": args.reason,
            "passes_required_criteria_dev_vs": BASELINES, "holdout_opened_before_choice": bool(args.holdout),
            "definition": next((r["definition"] for r in REGISTRY if r["candidate"] == c), None), "alpha": ALPHA_PRIMARY,
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, ensure_ascii=False), encoding="utf-8")
        print("final_choice.json escrito:", c)
    (OUT_DIR / "analysis_run.json").write_text(json.dumps({"folds": folds, "holdout_opened": args.holdout, "candidates": cands, "criteria": CRITERIA,
                                                          "baselines": BASELINES, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    pl.Config.set_tbl_width_chars(240); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(60)
    print(pooled.select("candidate", "set", "rmse", "mae", "medae", "cov_le300", "rmse_class0", "rmse_class1", "rmse_class2", "sse_share_class0", "sse_share_class1", "sse_share_class2", "n_pred_negative"))
    print(boot.select("candidate", "vs", "rmse_baseline", "rmse_candidate", "delta_rmse_point", "delta_p2.5", "delta_p97.5", "p_improve"))
    print(sel.select("candidate", "vs", "rmse_gain_s", "cov5_gain_pp", "folds_not_worse", "worst_fold_worsening_pct", "rmse_normal_gain_s", "share_gain_top50_rows", "p_improve_boot", "all_required_criteria"))
    print("decisión (desarrollo):", json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
