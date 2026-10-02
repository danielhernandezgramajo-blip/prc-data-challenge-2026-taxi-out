"""
V4 — Oráculos (cotas optimistas) del Gate A y escalera suplementaria, con IC bootstrap y desglose por aeropuerto.

    python src/v4_oracles.py

NO entrena y NO forma parte de ninguna candidata. Usa las predicciones OOF ya guardadas de V3 (S4A_mix; folds de origen
creciente julio y diciembre 2025; diciembre = proxy de enero; pesos del ranking) y los timestamps visibles de los Parquet de
training (MVT_ID solo para alinear filas). Cada oráculo es la mezcla de expertos de protocol.md §4 con una compuerta PERFECTA
(conoce el target): mide la ganancia máxima alcanzable, no una ganancia realizable.

Escalera (B→E es acumulativa; A2_* son variantes del Gate A):
  A2_nonSCHED_tau{10,30,60}: filas con un ancla no-SCHED a ±τ de BLOCK que no son copia de SCHED → MVT − ancla; resto V3.
                     τ = 10 s es el Gate A prerregistrado; 30/60 s son sensibilidad (cada vez menos "copia" y más "cercanía").
  B_all_anchors:     filas con CUALQUIER ancla (incl. SCHED) a ±10 s → MVT − ancla; resto V3.
  C_dayshift:        + clase de fecha desplazada (regime_class 2) perfecta → 86 400 + rama normal de V3.
  C2_rest_normal:    + resto predicho por la rama normal pura de V3 (sin la masa p1 que la mezcla reparte por prudencia).
  D_stage3_bulk:     + ganancia de capacidad/escala de la Etapa 3 (razón de MSE medida en el grueso de julio, exp_capacity.json)
                     aplicada al grueso (y ≤ 1 h) del resto. APROXIMACIÓN.
  E_perfect_tail:    + cola (y > 1 h) del resto perfecta.
Salidas: outputs/experiments/v4/oracles.json, outputs/experiments/v4/oracle_by_airport.csv
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

from config import OUTPUTS_DIR, SEED  # noqa: E402
from io_utils import scan_training  # noqa: E402
from v4_sources import ANCHORS, align_by_id, anchor_frame, anchor_matrix, expected_block  # noqa: E402

OUT = OUTPUTS_DIR / "experiments" / "v4"
W = {"Jul": 192_122 / 344_841, "Dec": 152_719 / 344_841}      # pesos del ranking; diciembre = proxy de enero
N_BOOT = 1000
COLS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt"]
STEPS = ["V3_S4A_mix", "A2_nonSCHED_tau10", "A2_nonSCHED_tau30", "A2_nonSCHED_tau60", "B_all_anchors", "C_dayshift",
         "C2_rest_normal", "D_stage3_bulk", "E_perfect_tail"]


def fold_inputs(fold: str, af: pl.DataFrame) -> dict:
    from v3_analysis import candidates, load_fold
    d = load_fold(fold)
    c = candidates(d)
    j = align_by_id(d["ids"], af)
    return {"y": d["y"].astype(np.float64), "v3": np.maximum(c["S4A_mix"], 0.0), "ts": anchor_matrix(j),
            "tm": j["t_mvt"].to_numpy().astype(np.float64),
            "normal": 0.5 * d["A_surface__direct_normal"] + 0.5 * d["A_surface__residual_normal"],
            "cls": d["regime_class"].astype(int), "airport": d["airport"].astype(str), "day": d["day"].astype(str)}


def gate_pred(f: dict, hit: np.ndarray, k_idx: np.ndarray, base: np.ndarray, p_ds: np.ndarray | None = None) -> np.ndarray:
    """Mezcla con compuerta perfecta (one-hot): ancla k_idx en las filas `hit`; experto 'none' = `base` en el resto;
    p_ds (0/1) activa el experto de fecha desplazada (none − 86 400)."""
    n, K = f["ts"].shape
    p = np.zeros((n, K + 1)); p[:, K] = 1.0
    rows = np.flatnonzero(hit)
    p[rows, K] = 0.0; p[rows, k_idx[rows]] = 1.0
    blk = expected_block(p, f["ts"], f["tm"] - base, p_ds=p_ds)
    return np.maximum(f["tm"] - blk, 0.0)


def ladder(f: dict, ratio_bulk: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Errores cuadráticos por paso y máscara de las filas que cada paso modifica."""
    y, ts, tm, v3 = f["y"], f["ts"], f["tm"], f["v3"]
    n = len(y); r = np.arange(n)
    dd = np.where(np.isnan(ts), np.inf, np.abs((tm - y)[:, None] - ts))
    sched_copy = dd[:, ANCHORS.index("SCHED")] <= 10
    others = np.array([i for i, a in enumerate(ANCHORS) if a != "SCHED"])
    best_o = others[np.argmin(dd[:, others], axis=1)]; d_o = dd[r, best_o]
    err = {"V3_S4A_mix": (v3 - y) ** 2}; pct = {}   # pct: máscaras de filas modificadas por cada paso
    for tau in (10, 30, 60):
        hit = (d_o <= tau) & ~sched_copy
        err[f"A2_nonSCHED_tau{tau}"] = (gate_pred(f, hit, best_o, v3) - y) ** 2; pct[f"A2_nonSCHED_tau{tau}"] = hit
    best = np.argmin(dd, axis=1); hit_all = dd[r, best] <= 10
    err["B_all_anchors"] = (gate_pred(f, hit_all, best, v3) - y) ** 2; pct["B_all_anchors"] = hit_all
    m2 = (f["cls"] == 2) & ~hit_all
    base = np.where(m2, f["normal"], v3); p_ds = m2.astype(np.float64)
    err["C_dayshift"] = (gate_pred(f, hit_all, best, base, p_ds) - y) ** 2; pct["C_dayshift"] = m2
    rest = ~hit_all & ~m2
    base2 = np.where(rest, np.maximum(f["normal"], 0.0), base)
    err["C2_rest_normal"] = (gate_pred(f, hit_all, best, base2, p_ds) - y) ** 2; pct["C2_rest_normal"] = rest
    e = err["C2_rest_normal"].copy(); mb = rest & (y <= 3600); e[mb] *= ratio_bulk
    err["D_stage3_bulk"] = e; pct["D_stage3_bulk"] = mb
    e = e.copy(); mt = rest & (y > 3600); e[mt] = 0.0
    err["E_perfect_tail"] = e; pct["E_perfect_tail"] = mt
    return err, pct


def _rows_info(e_v3: np.ndarray, mask: np.ndarray | None) -> dict | None:
    """Filas que modifica un paso: cuántas, RMSE de V3 en ellas y su cuota del SSE de V3 (explica cuánto puede ganar)."""
    if mask is None:
        return None
    return {"n": int(mask.sum()), "pct": 100 * float(mask.mean()),
            "v3_rmse_on_rows": float(np.sqrt(e_v3[mask].mean())) if mask.any() else None,
            "v3_sse_share_pct": 100 * float(e_v3[mask].sum() / e_v3.sum())}


def weighted_bootstrap(errs: dict, blocks: dict, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap pareado por bloques aeropuerto-día, estratificado por mes (cada fold se remuestrea por separado) y
    recombinado con los pesos del ranking. Mismos bloques para todos los pasos en cada réplica."""
    rng = np.random.default_rng(seed)
    prep = {}
    for fold in W:
        _, inv = np.unique(blocks[fold], return_inverse=True); nb = inv.max() + 1
        prep[fold] = (nb, np.bincount(inv, minlength=nb).astype(np.float64),
                      {s: np.bincount(inv, weights=errs[fold][s], minlength=nb) for s in STEPS})
    out = {s: np.empty(n_boot) for s in STEPS}
    for b in range(n_boot):
        mse = {s: 0.0 for s in STEPS}
        for fold, (nb, cnt, sums) in prep.items():
            idx = rng.integers(0, nb, nb); n = cnt[idx].sum()
            for s in STEPS:
                mse[s] += W[fold] * sums[s][idx].sum() / n
        for s in STEPS:
            out[s][b] = np.sqrt(mse[s])
    return out


def main() -> None:
    cap = json.loads((OUTPUTS_DIR / "experiments" / "v3" / "exp_capacity.json").read_text())
    ratio_bulk = (cap["C_cap+escala"]["rmse_grueso_le1h"] / cap["A_actual"]["rmse_grueso_le1h"]) ** 2
    af = anchor_frame(scan_training(COLS).filter(pl.col("PHASE_mvt") == "DEP").collect())
    errs, pcts, blocks, apts = {}, {}, {}, {}
    for fold in W:
        f = fold_inputs(fold, af)
        errs[fold], pcts[fold] = ladder(f, ratio_bulk)
        blocks[fold] = np.char.add(np.char.add(f["airport"], "|"), f["day"]); apts[fold] = f["airport"]
    point = {s: float(np.sqrt(sum(W[fd] * errs[fd][s].mean() for fd in W))) for s in STEPS}
    boot = weighted_bootstrap(errs, blocks, N_BOOT, SEED)
    res = {}
    for s in STEPS:
        dl = boot[s] - boot["V3_S4A_mix"]
        res[s] = {"rmse_w": point[s], "rmse_jul": float(np.sqrt(errs["Jul"][s].mean())), "rmse_dec": float(np.sqrt(errs["Dec"][s].mean())),
                  "delta_w_vs_v3": point[s] - point["V3_S4A_mix"],
                  "ci95_rmse_w": [float(np.quantile(boot[s], 0.025)), float(np.quantile(boot[s], 0.975))],
                  "ci95_delta_w": [float(np.quantile(dl, 0.025)), float(np.quantile(dl, 0.975))],
                  "rows_modified": {fd: _rows_info(errs[fd]["V3_S4A_mix"], pcts[fd].get(s)) for fd in W}}
    # coherencia con el Gate A (oracle_a2 de v4_audit.py)
    gate = json.loads((OUT / "gate_a.json").read_text())
    g_or = gate["A2_oracle"]["weighted"]["rmse_oracle"]
    assert abs(g_or - point["A2_nonSCHED_tau10"]) < 0.05, (g_or, point["A2_nonSCHED_tau10"])
    # por aeropuerto (RMSE ponderado jul/dic dentro de cada aeropuerto)
    rows = []
    for apt in sorted(set(apts["Jul"]) | set(apts["Dec"])):
        row = {"airport": apt, "n_jul": int((apts["Jul"] == apt).sum()), "n_dec": int((apts["Dec"] == apt).sum())}
        for s in ("V3_S4A_mix", "A2_nonSCHED_tau10", "B_all_anchors", "C2_rest_normal"):
            mj = errs["Jul"][s][apts["Jul"] == apt].mean(); md = errs["Dec"][s][apts["Dec"] == apt].mean()
            row[f"rmse_w__{s}"] = float(np.sqrt(W["Jul"] * mj + W["Dec"] * md))
        for s in ("A2_nonSCHED_tau10", "B_all_anchors", "C2_rest_normal"):
            row[f"delta__{s}"] = row[f"rmse_w__{s}"] - row["rmse_w__V3_S4A_mix"]
        rows.append(row)
    by_apt = pl.from_dicts(rows)
    by_apt.write_csv(OUT / "oracle_by_airport.csv")
    out = {"description": (__doc__ or "").strip(), "folds": "Jul 2025 y Dec 2025 (proxy de enero), OOF origen creciente de V3",
           "weights": W, "n_boot": N_BOOT, "seed": SEED, "bootstrap": "pareado, bloques aeropuerto-día, estratificado por mes",
           "stage3_bulk_mse_ratio": ratio_bulk, "steps": res,
           "airports_A2_tau10_delta_le_-15s": int((by_apt["delta__A2_nonSCHED_tau10"] <= -15).sum()),
           "airports_total": by_apt.height}
    (OUT / "oracles.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for s in STEPS:
        r = res[s]
        print(f"{s:20s} RMSE_w {r['rmse_w']:6.1f} [{r['ci95_rmse_w'][0]:6.1f}, {r['ci95_rmse_w'][1]:6.1f}]  Δ {r['delta_w_vs_v3']:+6.1f} "
              f"[{r['ci95_delta_w'][0]:+6.1f}, {r['ci95_delta_w'][1]:+6.1f}]  Jul {r['rmse_jul']:6.1f} Dec {r['rmse_dec']:6.1f}")
    pl.Config.set_tbl_width_chars(200); pl.Config.set_tbl_cols(20)
    print(by_apt.select("airport", "rmse_w__V3_S4A_mix", "delta__A2_nonSCHED_tau10", "delta__B_all_anchors", "delta__C2_rest_normal"))


if __name__ == "__main__":
    main()
