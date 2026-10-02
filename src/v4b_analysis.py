"""
V4b — Métricas, bootstrap, diagnóstico de clasificadores y regla de decisión (protocol.md §7–8).

    python src/v4b_analysis.py

Lee las predicciones outer de la etapa 2 (models/experiments/v4b/stage2/) y la selección anidada (stage2_selection.json).
Métrica primaria: RMSE_w del ensamble de las semillas disponibles (media de predicciones). Salidas en outputs/experiments/v4b/:
metrics.json, metrics_by_candidate.csv, breakdown.csv, by_airport.csv, classifier_diag.csv, decision.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from v4b_protocol import OUT_DIR, OUTER, PRED_DIR, SEEDS, pred_path  # noqa: E402
from v4b_stage2 import ST2_DIR, available, load_part  # noqa: E402

W = {"LOJO": 152_719 / 344_841, "LOJuO": 192_122 / 344_841}      # enero / julio (pesos del ranking)
MONTH_NAME = {"LOJO": "ene", "LOJuO": "jul"}
THRESHOLD_S = -15.0
N_BOOT = 1000


def rmse_w(mse: dict[str, float]) -> float:
    return float(np.sqrt(sum(W[f] * mse[f] for f in W)))


def seeds_complete() -> list[int]:
    return [s for s in SEEDS if all(available(f, s) for f in OUTER) and all((ST2_DIR / f"C0__{f}__s{s}__outer.npy").exists() for f in OUTER)]


def load_pred(key: str, fold: str, seed: int) -> np.ndarray:
    return np.load(ST2_DIR / f"{key}__{fold}__s{seed}__outer.npy").astype(np.float64)


def ensemble(key: str, fold: str, seeds: list[int]) -> np.ndarray:
    return np.mean([load_pred(key, fold, s) for s in seeds], axis=0)


def all_keys() -> list[str]:
    return sorted({p.name.split("__")[0] if p.name.startswith("C0__") else "__".join(p.name.split("__")[:4])
                   for p in ST2_DIR.glob("*__outer.npy")})


def breakdown(y: np.ndarray, pred: np.ndarray, cls: np.ndarray) -> dict:
    e = (pred - y) ** 2; sse = e.sum()
    groups = {"total": np.ones_like(y, bool), "clase0": cls == 0, "clase0_le1h": (cls == 0) & (y <= 3600),
              "clase0_1a3h": (cls == 0) & (y > 3600) & (y <= 10800), "clase0_gt3h": (cls == 0) & (y > 10800),
              "clase1": cls == 1, "clase2": cls == 2}
    out = {}
    for k, m in groups.items():
        out[f"n_{k}"] = int(m.sum())
        out[f"rmse_{k}"] = float(np.sqrt(e[m].mean())) if m.any() else None
        out[f"sse_share_{k}"] = float(100 * e[m].sum() / sse) if k != "total" else 100.0
    out["mae_total"] = float(np.abs(pred - y).mean())
    return out


def weighted_block_bootstrap(errs: dict[str, dict[str, np.ndarray]], blocks: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    """errs[fold][name] = errores cuadráticos por fila. Remuestrea bloques aeropuerto-día dentro de cada mes; misma muestra para
    todos los nombres. Devuelve RMSE_w por réplica para cada nombre."""
    rng = np.random.default_rng(seed)
    names = list(next(iter(errs.values())).keys())
    prep = {}
    for f in W:
        _, inv = np.unique(blocks[f], return_inverse=True); nb = inv.max() + 1
        prep[f] = (nb, np.bincount(inv, minlength=nb).astype(np.float64), {k: np.bincount(inv, weights=errs[f][k], minlength=nb) for k in names})
    out = {k: np.empty(n_boot) for k in names}
    for i in range(n_boot):
        mse = {k: 0.0 for k in names}
        for f, (nb, cnt, sums) in prep.items():
            idx = rng.integers(0, nb, nb); n = cnt[idx].sum()
            for k in names:
                mse[k] += W[f] * sums[k][idx].sum() / n
        for k in names:
            out[k][i] = np.sqrt(mse[k])
    return out


def auc(y1: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y1, p)) if 0 < y1.sum() < y1.size else float("nan")


def classifier_diag(fold: str, seeds: list[int]) -> list[dict]:
    rows = []
    for s in seeds:
        d = load_part(fold, s, "outer")
        m = ~d["in_cell"]; y1 = (d["regime_class"][m] == 1).astype(float)
        delta2 = (d["s"][m] - d["b0"][m]) ** 2
        for name, p in (("p1_C0 (V3)", d["P0"][m, 1]), ("p1_new (V4b)", d["p1n"][m])):
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            rows.append({"fold": fold, "mes": MONTH_NAME[fold], "seed": s, "classifier": name, "auc": auc(y1, p),
                         "logloss": float(-(y1 * np.log(pc) + (1 - y1) * np.log(1 - pc)).mean()),
                         "brier": float(((p - y1) ** 2).mean()), "brier_w_delta2": float((delta2 * (p - y1) ** 2).sum() / delta2.sum()),
                         "mean_p_class1": float(p[y1 == 1].mean()), "mean_p_class0": float(p[y1 == 0].mean())})
    return rows


def normal_branch_rows(seeds: list[int]) -> list[dict]:
    """RMSE de cada rama normal en filas de clase 0 del mes outer (sin compuerta): responde "¿rama normal insuficiente?"."""
    rows = []
    for f in OUTER:
        ens = {}
        for s in seeds:
            d = load_part(f, s, "outer")
            m0 = d["regime_class"] == 0; le1 = m0 & (d["y"] <= 3600)
            for name, b in (("normal_C0 (V3)", d["b0"]), ("normal_C2 (V4b)", d["b2"]), ("blend 50/50", 0.5 * (d["b0"] + d["b2"]))):
                ens.setdefault(name, []).append(b)
                rows.append({"fold": f, "mes": MONTH_NAME[f], "seed": str(s), "normal": name,
                             "rmse_clase0": float(np.sqrt(((np.maximum(b, 0) - d["y"])[m0] ** 2).mean())),
                             "rmse_clase0_le1h": float(np.sqrt(((np.maximum(b, 0) - d["y"])[le1] ** 2).mean()))})
        for name, lst in ens.items():
            b = np.mean(lst, axis=0)
            rows.append({"fold": f, "mes": MONTH_NAME[f], "seed": "ens", "normal": name,
                         "rmse_clase0": float(np.sqrt(((np.maximum(b, 0) - d["y"])[m0] ** 2).mean())),
                         "rmse_clase0_le1h": float(np.sqrt(((np.maximum(b, 0) - d["y"])[le1] ** 2).mean()))})
    return rows


def ablation_ctr2() -> dict | None:
    """Ablación A1: rama normal con max_ctr_complexity 2 frente a 1 (LOJO, semilla 42), RMSE de clase 0."""
    fold, seed = "LOJO", 42
    if not all(pred_path(c, fold, seed, "outer").exists() for c in ("c2_n", "c2_n_ctr2")):
        return None
    d = load_part(fold, seed, "outer"); m0 = d["regime_class"] == 0; le1 = m0 & (d["y"] <= 3600)
    out = {"fold": fold, "seed": seed}
    for name, comp in (("ctr1", "c2_n"), ("ctr2", "c2_n_ctr2")):
        b = np.maximum(np.load(pred_path(comp, fold, seed, "outer")).astype(np.float64), 0)
        out[f"rmse_clase0_{name}"] = float(np.sqrt(((b - d["y"])[m0] ** 2).mean()))
        out[f"rmse_clase0_le1h_{name}"] = float(np.sqrt(((b - d["y"])[le1] ** 2).mean()))
    out["delta_clase0"] = out["rmse_clase0_ctr2"] - out["rmse_clase0_ctr1"]
    return out


def row_influence(errs: dict, meta: dict, a: str, b: str, k: int = 3) -> dict:
    """Influencia de filas individuales en Δ = RMSE_w(a) − RMSE_w(b): para las k filas de cada mes con mayor |e_a − e_b|,
    Δ recalculado sin esa fila. Diagnóstico de robustez (no decide)."""
    base = rmse_w({f: float(errs[f][a].mean()) for f in OUTER}) - rmse_w({f: float(errs[f][b].mean()) for f in OUTER})
    rows = []
    for f in OUTER:
        d = errs[f][a] - errs[f][b]
        for i in np.argsort(-np.abs(d))[:k]:
            keep = np.ones(d.size, bool); keep[i] = False
            mse = {g: (errs[g][a][keep].mean() if g == f else errs[g][a].mean(), errs[g][b][keep].mean() if g == f else errs[g][b].mean()) for g in OUTER}
            dw = rmse_w({g: float(v[0]) for g, v in mse.items()}) - rmse_w({g: float(v[1]) for g, v in mse.items()})
            rows.append({"mes": MONTH_NAME[f], "airport": str(meta[f]["airport"][i]), "y": float(meta[f]["y"][i]), "s": float(meta[f]["s"][i]),
                         "clase": int(meta[f]["regime_class"][i]), "d_sse_row": float(d[i]), "share_of_month_d_sse": float(d[i] / d.sum()) if d.sum() else None,
                         "delta_w_without_row": float(dw)})
    return {"delta_w": float(base), "rows": rows}


def main() -> None:
    seeds = seeds_complete()
    assert seeds, "no hay semillas completas"
    sel = json.loads((OUT_DIR / "stage2_selection.json").read_text())
    prim = {"C0": "C0", "C1": sel["C1"]["key"], "C2": sel["C2"]["key"]}
    meta = {f: dict(np.load(PRED_DIR / f"meta__{f}__outer.npz", allow_pickle=False)) for f in OUTER}
    for f in OUTER:
        meta[f]["y"] = meta[f]["y"].astype(np.float64)
    # --- todas las candidatas (diagnóstico) y primarias -----------------------------------------------------------
    keys = all_keys()
    cand_rows, ens = [], {}
    for k in keys:
        mse_ens = {}
        for f in OUTER:
            ens[(k, f)] = ensemble(k, f, seeds)
            mse_ens[f] = float(((ens[(k, f)] - meta[f]["y"]) ** 2).mean())
        per_seed = [rmse_w({f: float(((load_pred(k, f, s) - meta[f]["y"]) ** 2).mean()) for f in OUTER}) for s in seeds]
        cand_rows.append({"key": k, "rmse_w_ens": rmse_w(mse_ens), "rmse_ene_ens": float(np.sqrt(mse_ens["LOJO"])),
                          "rmse_jul_ens": float(np.sqrt(mse_ens["LOJuO"])), "rmse_w_seed_min": min(per_seed), "rmse_w_seed_max": max(per_seed),
                          "primary": next((c for c, v in prim.items() if v == k), "")})
    cand = pl.from_dicts(cand_rows).sort("rmse_w_ens")
    c0w = cand.filter(pl.col("key") == "C0")["rmse_w_ens"][0]
    cand = cand.with_columns((pl.col("rmse_w_ens") - c0w).alias("delta_vs_C0"))
    cand.write_csv(OUT_DIR / "metrics_by_candidate.csv")
    # --- desglose de las primarias --------------------------------------------------------------------------------
    bd_rows, apt_rows = [], []
    for c, k in prim.items():
        for f in OUTER:
            y, cls = meta[f]["y"], meta[f]["regime_class"]
            bd_rows.append({"config": c, "key": k, "fold": f, "mes": MONTH_NAME[f], **breakdown(y, ens[(k, f)], cls)})
            for apt in np.unique(meta[f]["airport"]):
                m = meta[f]["airport"] == apt
                apt_rows.append({"config": c, "fold": f, "airport": apt, "n": int(m.sum()), "mse": float(((ens[(k, f)][m] - y[m]) ** 2).mean())})
    pl.from_dicts(bd_rows, infer_schema_length=None).write_csv(OUT_DIR / "breakdown.csv")
    apt = pl.from_dicts(apt_rows)
    apt_w = (apt.with_columns(pl.col("fold").replace_strict(W, return_dtype=pl.Float64).alias("w"))
                .group_by("config", "airport").agg((pl.col("w") * pl.col("mse")).sum().sqrt().alias("rmse_w")).sort("airport"))
    apt_piv = apt_w.pivot(on="config", index="airport", values="rmse_w").with_columns(
        (pl.col("C1") - pl.col("C0")).alias("d_C1_C0"), (pl.col("C2") - pl.col("C0")).alias("d_C2_C0"))
    apt_piv.write_csv(OUT_DIR / "by_airport.csv")
    # --- bootstrap ---------------------------------------------------------------------------------------------------
    errs = {f: {c: (ens[(k, f)] - meta[f]["y"]) ** 2 for c, k in prim.items()} for f in OUTER}
    blocks = {f: np.char.add(np.char.add(meta[f]["airport"], "|"), meta[f]["day"]) for f in OUTER}
    boot = weighted_block_bootstrap(errs, blocks, N_BOOT, 42)
    point = {c: rmse_w({f: float(errs[f][c].mean()) for f in OUTER}) for c in prim}
    comps = {"C1-C0": ("C1", "C0"), "C2-C0": ("C2", "C0"), "C2-C1": ("C2", "C1")}
    deltas = {}
    for name, (a, b) in comps.items():
        dl = boot[a] - boot[b]
        per_seed = []
        for s in seeds:
            ra = rmse_w({f: float(((load_pred(prim[a], f, s) - meta[f]["y"]) ** 2).mean()) for f in OUTER})
            rb = rmse_w({f: float(((load_pred(prim[b], f, s) - meta[f]["y"]) ** 2).mean()) for f in OUTER})
            per_seed.append(ra - rb)
        by_month = {MONTH_NAME[f]: float(np.sqrt(errs[f][a].mean()) - np.sqrt(errs[f][b].mean())) for f in OUTER}
        deltas[name] = {"delta_w": point[a] - point[b], "ci95": [float(np.quantile(dl, 0.025)), float(np.quantile(dl, 0.975))],
                        "p_improve": float((dl < 0).mean()), "per_seed": per_seed, "by_month": by_month}
    # --- clasificadores -----------------------------------------------------------------------------------------------
    diag = pl.from_dicts([r for f in OUTER for r in classifier_diag(f, seeds)])
    diag.write_csv(OUT_DIR / "classifier_diag.csv")
    # sensibilidad (NO decide): sin la cola de clase 0 > 3 h (filas que ningún modelo predice; bin prefijado en §7)
    keep = {f: ~((meta[f]["regime_class"] == 0) & (meta[f]["y"] > 10800)) for f in OUTER}
    p_ex = {c: rmse_w({f: float(errs[f][c][keep[f]].mean()) for f in OUTER}) for c in prim}
    sens = {"excluded_rows": {MONTH_NAME[f]: int((~keep[f]).sum()) for f in OUTER}, "rmse_w": p_ex,
            "delta_C1_C0": p_ex["C1"] - p_ex["C0"], "delta_C2_C0": p_ex["C2"] - p_ex["C0"], "delta_C2_C1": p_ex["C2"] - p_ex["C1"]}
    infl = {name: row_influence(errs, meta, a, b) for name, (a, b) in comps.items()}
    nb = pl.from_dicts(normal_branch_rows(seeds)); nb.write_csv(OUT_DIR / "normal_branch.csv")
    abl = ablation_ctr2()
    # --- decisión -------------------------------------------------------------------------------------------------------
    d20 = deltas["C2-C0"]
    passes = d20["delta_w"] <= THRESHOLD_S
    fragile = passes and (d20["ci95"][1] > 0 or max(d20["per_seed"]) > 0)
    decision = {"seeds": seeds, "preliminary": len(seeds) < len(SEEDS), "primary": prim, "rmse_w": point, "deltas": deltas,
                "ablation_ctr2": abl, "sensitivity_excl_class0_gt3h": sens, "single_row_influence": infl,
                "threshold_s": THRESHOLD_S, "pass_to_costly_stage": bool(passes), "fragile": bool(fragile),
                "candidate_generated": False, "uploaded_to_bucket": False, "pushed": False, "leaderboard_used": False,
                "note_oracles": "Las cotas de oráculo de V4 (230.2, 205.7) no son alcanzables y no se usan como referencia."}
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"rmse_w_primary": point, "deltas": deltas, "seeds": seeds}, indent=2), encoding="utf-8")
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(40)
    print(cand.head(40))
    print(json.dumps(decision, indent=1, ensure_ascii=False))
    print(nb.filter(pl.col("seed") == "ens"))
    print(diag.group_by("mes", "classifier").agg(pl.col("auc", "logloss", "brier_w_delta2", "mean_p_class1", "mean_p_class0").mean()).sort("mes", "classifier"))


if __name__ == "__main__":
    main()
