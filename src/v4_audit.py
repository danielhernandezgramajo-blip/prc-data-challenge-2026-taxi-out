"""
V4 — Etapa A: auditoría de las fuentes visibles de BLOCK_TIME_UTC_mvt sobre las salidas de training 2025, y Gate A.

    python src/v4_audit.py

Salidas (outputs/experiments/v4/): source_coverage.csv, source_attribution_by_airport.csv, source_attribution_by_month.csv,
source_confusion.csv, source_attribution_by_group.csv, source_group_informativeness.csv, timestamp_resolution.csv,
aux_stand_anchors.csv, gate_a.json. Solo lectura de los Parquet. No entrena modelos. Los informes (source_audit.md,
report.md, decision.json) los genera src/v4_report.py a partir de estas salidas y de src/v4_oracles.py.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from config import OUTPUTS_DIR  # noqa: E402
from io_utils import scan_training  # noqa: E402
from v4_sources import ANCHORS, ANCHOR_SRC, SOURCES, TOLS, align_by_id, anchor_frame, anchor_matrix, attribute, copy_mass  # noqa: E402

OUT = OUTPUTS_DIR / "experiments" / "v4"
TAU = 30.0                      # tolerancia de atribución prefijada en protocol.md
W_JAN, W_JUL = 152_719 / 344_841, 192_122 / 344_841
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "ADEP_flt", "ADES_flt",
        "ADES_FILED_flt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt",
        "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = scan_training(COLS).collect()
    dep = raw.filter(pl.col("PHASE_mvt") == "DEP").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR")
    return dep, arr


def coverage_table(ts: np.ndarray, block: np.ndarray, airports: np.ndarray) -> pl.DataFrame:
    n = len(block); rows = []
    for j, a in enumerate(ANCHORS):
        d = block - ts[:, j]; avail = np.isfinite(d); ad = np.abs(d)
        # corrección residual: centrar con la moda por aeropuerto (bins de 1 s dentro de ±120 s)
        dc = d.copy()
        for apt in np.unique(airports):
            m = (airports == apt) & avail & (ad <= 120)
            if m.sum() >= 200:
                vals, cnt = np.unique(np.round(d[m]), return_counts=True); dc[airports == apt] -= vals[np.argmax(cnt)]
        row = {"anchor": a, "source": ANCHOR_SRC[a][0], "shift_s": ANCHOR_SRC[a][1], "n_available": int(avail.sum()),
               "pct_available": 100 * avail.mean()}
        for t in TOLS:
            row[f"pct_within_{t}s"] = 100 * float((ad <= t).sum()) / n
        row["copy_mass_pct"] = 100 * copy_mass(d, n)
        row["copy_mass30_pct"] = 100 * copy_mass(d, n, inner=30, sh_lo=60, sh_hi=150)
        row["copy_mass_corrected_pct"] = 100 * copy_mass(dc, n)
        row["median_d_within_120"] = float(np.median(d[avail & (ad <= 120)])) if (avail & (ad <= 120)).any() else np.nan
        rows.append(row)
    # uniones
    def union(cols, t):
        return 100 * float((np.nanmin(np.where(np.isnan(ts[:, cols]), np.inf, np.abs(block[:, None] - ts[:, cols])), axis=1) <= t).sum()) / n
    idx_all = list(range(len(ANCHORS))); idx_nosched = [i for i, a in enumerate(ANCHORS) if a != "SCHED"]
    idx_aobt = [i for i, a in enumerate(ANCHORS) if a.startswith("AOBT3")]
    for name, cols in (("UNION_all", idx_all), ("UNION_all_but_SCHED", idx_nosched), ("UNION_AOBT3", idx_aobt)):
        row = {"anchor": name, "source": "union", "shift_s": np.nan, "n_available": n, "pct_available": 100.0}
        for t in TOLS:
            row[f"pct_within_{t}s"] = union(cols, t)
        rows.append(row)
    return pl.from_dicts(rows, infer_schema_length=None)


def attribution_shares(cls: np.ndarray, keys: np.ndarray, keyname: str) -> pl.DataFrame:
    lab = np.where(cls >= 0, np.array(ANCHORS + ["none"], dtype=object)[np.where(cls >= 0, cls, len(ANCHORS))], "none")
    df = pl.DataFrame({keyname: keys, "cls": lab.astype(str)})
    tot = df.group_by(keyname).len().rename({"len": "n"})
    piv = df.group_by(keyname, "cls").len().pivot(on="cls", index=keyname, values="len").fill_null(0)
    out = tot.join(piv, on=keyname)
    cls_cols = [c for c in ANCHORS + ["none"] if c in out.columns]       # orden fijo de columnas (el pivot no lo garantiza)
    out = out.select(keyname, "n", *cls_cols)
    for c in cls_cols:
        out = out.with_columns((100 * pl.col(c) / pl.col("n")).round(3).alias(c))
    return out.sort(keyname)


def confusion(ts: np.ndarray, block: np.ndarray, tol: float) -> pl.DataFrame:
    within = np.abs(block[:, None] - ts) <= tol
    rows = []
    for i, a in enumerate(ANCHORS):
        wi = within[:, i]; ni = int(wi.sum())
        for j, b in enumerate(ANCHORS):
            if ni == 0:
                continue
            same_ts = np.abs(ts[:, i] - ts[:, j]) <= 1
            rows.append({"anchor_A": a, "anchor_B": b, "n_A_within": ni, "pct_B_within_given_A": 100 * float((wi & within[:, j]).sum()) / ni,
                         "pct_ts_equal_given_A": 100 * float((wi & same_ts).sum()) / ni})
    return pl.from_dicts(rows)


def out_of_month_rate(inv: np.ndarray, month: np.ndarray, y: np.ndarray, prior: float, k: float = 20.0) -> np.ndarray:
    """Tasa del grupo de cada fila estimada SOLO con filas de otros meses ("fuera de mes"), suavizada hacia `prior` (una
    constante común a todas las filas) con pseudo-recuento k. Ninguna fila ve el target de su propio mes."""
    inv = np.asarray(inv, dtype=np.int64); mi = np.asarray(month, dtype=np.int64)
    G = int(inv.max()) + 1; mi = mi - mi.min(); M = int(mi.max()) + 1
    key = inv * M + mi
    s_gm = np.bincount(key, weights=y, minlength=G * M); n_gm = np.bincount(key, minlength=G * M).astype(np.float64)
    s_g = np.bincount(inv, weights=y, minlength=G); n_g = np.bincount(inv, minlength=G).astype(np.float64)
    return (s_g[inv] - s_gm[key] + k * prior) / (n_g[inv] - n_gm[key] + k)


def group_informativeness(dep: pl.DataFrame, targets: dict[str, np.ndarray]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Por cada agrupación visible en ranking: tasa por grupo de cada clase objetivo y AUC de la tasa del grupo estimada
    FUERA DE MES (sin ver el mes de la fila) para predecir la clase de la fila. Mide si la agrupación informa sobre la
    fuente de forma generalizable a un mes no visto. (Sustituye al AUC leave-one-out por fila, que en agrupaciones de baja
    cardinalidad con una categoría dominante queda sesgado a la baja —p. ej. 0.01— porque dentro de cada grupo los
    positivos reciben sistemáticamente una tasa menor que los negativos.) "month" da 0.5 por construcción."""
    groups = {
        "airport": pl.col("ADEP_mvt"),
        "month": pl.col("month").cast(pl.String),
        "operator": pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"),
        "flight_number": pl.col("FLIGHT_mvt").fill_null("NA"),
        "callsign": pl.col("CALLSIGN_flt").fill_null("NA"),
        "airport_stand": pl.col("ADEP_mvt") + "|" + pl.col("STAND_mvt").cast(pl.String).fill_null("NA"),
        "airport_runway": pl.col("ADEP_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.String).fill_null("NA"),
        "route": pl.col("ADEP_mvt") + ">" + pl.col("ADES_mvt").fill_null("NA"),
        "presence_pattern": pl.concat_str([pl.col(s).is_null().cast(pl.Int8).cast(pl.String) for s in SOURCES]),
        "aobt3_resolution": pl.col("ADEP_mvt") + "|" + pl.when(pl.col("AOBT_3_flt").is_null()).then(pl.lit("absent"))
                             .when(pl.col("AOBT_3_flt").dt.second() == 0).then(pl.lit("min")).otherwise(pl.lit("sec")),
    }
    keys = dep.select([e.alias(k) for k, e in groups.items()])
    month = dep["month"].to_numpy()
    info_rows, top_rows = [], []
    for gname in groups:
        g = keys[gname].to_numpy().astype(str)
        uniq, inv = np.unique(g, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        for tname, y in targets.items():
            y = y.astype(float); prior = float(y.mean())
            s = np.bincount(inv, weights=y)
            enc = out_of_month_rate(inv, month, y, prior)
            auc = roc_auc_score(y, enc) if 0 < y.sum() < len(y) else np.nan
            rate = s / cnt; big = cnt >= 200
            info_rows.append({"grouping": gname, "target": tname, "n_groups": int(len(uniq)), "n_groups_ge200": int(big.sum()),
                              "prior_pct": 100 * prior, "auc_out_of_month": auc,
                              "note": "0.5 por construccion (fuera de mes)" if gname == "month" else "",
                              "p10_rate_pct_ge200": 100 * float(np.quantile(rate[big], 0.1)) if big.any() else np.nan,
                              "p90_rate_pct_ge200": 100 * float(np.quantile(rate[big], 0.9)) if big.any() else np.nan,
                              "max_rate_pct_ge200": 100 * float(rate[big].max()) if big.any() else np.nan})
            if gname in ("airport", "month", "presence_pattern", "aobt3_resolution") or tname == "copy_any":
                order = np.argsort(-cnt)[:40]
                for k in order:
                    top_rows.append({"grouping": gname, "group": uniq[k] if gname not in ("operator",) else uniq[k][:12], "target": tname,
                                     "n": int(cnt[k]), "rate_pct": 100 * float(rate[k])})
    return pl.from_dicts(info_rows), pl.from_dicts(top_rows)


def resolution_table(dep: pl.DataFrame) -> pl.DataFrame:
    cols = ["BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]
    aggs = []
    for c in cols:
        aggs += [(pl.col(c).is_not_null().mean() * 100).round(2).alias(f"{c}__pct_present"),
                 ((pl.col(c).dt.second() == 0).sum() / pl.col(c).is_not_null().sum() * 100).round(2).alias(f"{c}__pct_sec0")]
    return dep.group_by("ADEP_mvt").agg(aggs).sort("ADEP_mvt")


def aux_stand_anchors(dep: pl.DataFrame, arr: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Anclas auxiliares derivadas (no son copias): on-block de la llegada siguiente / anterior al mismo (aeropuerto, stand),
    visibles en ranking. Se busca la llegada siguiente a partir de t_ref = coalesce(AOBT_3, MVT − 900 s) − 300 s."""
    a = (arr.filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
            .select(pl.col("ADES_mvt").alias("apt"), pl.col("STAND_mvt").cast(pl.String).fill_null("NA").alias("stand"),
                    pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_arr_onblock")).sort("t_arr_onblock"))
    d = dep.select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("apt"), pl.col("STAND_mvt").cast(pl.String).fill_null("NA").alias("stand"),
                   pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_block"),
                   pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_mvt"),
                   pl.col("AOBT_3_flt").dt.epoch("s").cast(pl.Float64).alias("t_aobt"))
    d = d.with_columns((pl.coalesce(pl.col("t_aobt"), pl.col("t_mvt") - 900.0) - 300.0).alias("t_ref")).sort("t_ref")
    nxt = d.join_asof(a.rename({"t_arr_onblock": "t_next_arr"}), left_on="t_ref", right_on="t_next_arr", by=["apt", "stand"],
                      strategy="forward", tolerance=6 * 3600.0)
    prv = d.join_asof(a.rename({"t_arr_onblock": "t_prev_arr"}), left_on="t_ref", right_on="t_prev_arr", by=["apt", "stand"],
                      strategy="backward", tolerance=24 * 3600.0)
    j = nxt.join(prv.select("MVT_ID_mvt", "t_prev_arr"), on="MVT_ID_mvt")
    g = (j["t_next_arr"] - j["t_block"]).to_numpy(); gp = (j["t_block"] - j["t_prev_arr"]).to_numpy()
    ok = np.isfinite(g)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    stats = {"pct_dep_with_next_arr_6h": 100 * float(ok.mean()),
             "next_gap_quantiles_min": [float(np.quantile(g[ok], q)) / 60 for q in qs],
             "pct_next_gap_0_10min": 100 * float(((g >= 0) & (g <= 600)).sum()) / len(g),
             "pct_next_gap_negative": 100 * float((g < 0).sum()) / len(g),
             "prev_gap_quantiles_min": [float(np.nanquantile(gp, q)) / 60 for q in qs]}
    per_apt = (j.with_columns((pl.col("t_next_arr") - pl.col("t_block")).alias("gap"))
                .group_by("apt").agg(pl.len().alias("n"), pl.col("gap").is_not_null().mean().round(3).alias("frac_with_next"),
                                     (pl.col("gap") / 60).median().round(1).alias("median_gap_min"),
                                     ((pl.col("gap") >= 0) & (pl.col("gap") <= 600)).mean().round(3).alias("frac_gap_0_10min"),
                                     (pl.col("gap") < 0).mean().round(3).alias("frac_gap_negative")).sort("apt"))
    return per_apt, stats


def oracle_a2(dep_af: pl.DataFrame) -> dict:
    """Cota optimista: en las filas de validación (julio y diciembre 2025, OOF de S4A_mix de V3) con un ancla no-SCHED dentro
    de ±10 s de BLOCK, sustituir la predicción por MVT − ancla."""
    from v3_analysis import candidates, load_fold
    res = {}
    for fold in ("Jul", "Dec"):
        d = load_fold(fold); c = candidates(d); y = d["y"]; p = c["S4A_mix"]
        j = align_by_id(d["ids"], dep_af)          # MVT_ID solo para alinear; error si falta o se duplica algún id
        ts = anchor_matrix(j)
        block = j["t_mvt"].to_numpy() - y
        dd = np.abs(block[:, None] - ts)
        sched_copy = dd[:, ANCHORS.index("SCHED")] <= 10
        others = [i for i, a in enumerate(ANCHORS) if a != "SCHED"]
        do = np.where(np.isnan(dd[:, others]), np.inf, dd[:, others]); best = np.argmin(do, axis=1)
        hit = (do[np.arange(len(y)), best] <= 10) & ~sched_copy
        p_or = p.copy(); p_or[hit] = j["t_mvt"].to_numpy()[hit] - ts[hit, np.array(others)[best[hit]]]
        mse0 = float(np.mean((np.maximum(p, 0) - y) ** 2)); mse1 = float(np.mean((np.maximum(p_or, 0) - y) ** 2))
        res[fold] = {"n": int(len(y)), "n_oracle_rows": int(hit.sum()), "pct_oracle_rows": 100 * float(hit.mean()),
                     "rmse_s4a": mse0 ** 0.5, "rmse_oracle": mse1 ** 0.5, "mse_s4a": mse0, "mse_oracle": mse1}
    w0 = (W_JUL * res["Jul"]["mse_s4a"] + W_JAN * res["Dec"]["mse_s4a"]) ** 0.5
    w1 = (W_JUL * res["Jul"]["mse_oracle"] + W_JAN * res["Dec"]["mse_oracle"]) ** 0.5
    res["weighted"] = {"rmse_s4a": w0, "rmse_oracle": w1, "delta": w1 - w0, "note": "diciembre como proxy de enero; pesos del ranking"}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dep, arr = load()
    print(f"DEP 2025: {dep.height:,} | ARR: {arr.height:,}", flush=True)
    af = anchor_frame(dep)
    ts = anchor_matrix(af)
    block = dep["BLOCK_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64).to_numpy()
    airports = dep["ADEP_mvt"].to_numpy().astype(str)

    cov = coverage_table(ts, block, airports); cov.write_csv(OUT / "source_coverage.csv")
    at = attribute(ts, block, TAU); at10 = attribute(ts, block, 10.0)
    attribution_shares(at["cls"], airports, "ADEP_mvt").write_csv(OUT / "source_attribution_by_airport.csv")
    attribution_shares(at["cls"], dep["month"].to_numpy(), "month").write_csv(OUT / "source_attribution_by_month.csv")
    confusion(ts, block, TAU).write_csv(OUT / "source_confusion.csv")
    resolution_table(dep).write_csv(OUT / "timestamp_resolution.csv")

    # resumen por fila (fuente más cercana, segunda, ambigüedad, ausencias, desplazamiento)
    near = np.array(ANCHORS + ["(none)"], dtype=object)[np.where(at["nearest"] >= 0, at["nearest"], len(ANCHORS))]
    row_summary = {
        "pct_rows_attributed_tau30": 100 * float((at["cls"] >= 0).mean()),
        "pct_rows_attributed_tau10": 100 * float((at10["cls"] >= 0).mean()),
        "pct_ambiguous_tau30": 100 * float(at["ambiguous"].mean()),
        "pct_attributed_to_shifted_anchor_tau30": 100 * float(np.isin(at["cls"], [i for i, a in enumerate(ANCHORS) if a.endswith("d")]).mean()),
        "nearest_anchor_share_pct": {a: 100 * float((near == a).mean()) for a in ANCHORS + ["(none)"]},
        "dist1_quantiles_s": [float(np.nanquantile(at["dist1"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
        "pct_rows_missing_source": {s: 100 * float(dep[s].is_null().mean()) for s in SOURCES},
    }
    # objetivos para la informatividad de agrupaciones (copias a ±10 s)
    sched_i = ANCHORS.index("SCHED"); aobt_i = ANCHORS.index("AOBT3")
    d10 = np.abs(block[:, None] - ts) <= 10
    other_plan = [ANCHORS.index(a) for a in ("IOBT", "LOBT", "EOBT1")]
    targets = {"copy_SCHED": d10[:, sched_i], "copy_AOBT3": d10[:, aobt_i] & ~d10[:, sched_i],
               "copy_IOBT_LOBT_EOBT_only": d10[:, other_plan].any(axis=1) & ~d10[:, sched_i] & ~d10[:, aobt_i],
               "copy_any": d10.any(axis=1)}
    info, top = group_informativeness(dep, targets)
    info.write_csv(OUT / "source_group_informativeness.csv"); top.write_csv(OUT / "source_attribution_by_group.csv")

    aux_apt, aux_stats = aux_stand_anchors(dep, arr); aux_apt.write_csv(OUT / "aux_stand_anchors.csv")

    # Gate A
    cm = {r["anchor"]: (r["copy_mass_pct"], r["copy_mass_corrected_pct"]) for r in cov.iter_rows(named=True) if r["source"] != "union"}
    # masa de copia sin doble conteo: copias a ±10 s de alguna ancla no-SCHED que NO son copia de SCHED, menos fondo
    nosched = [i for i, a in enumerate(ANCHORS) if a != "SCHED"]
    dmin_other = np.nanmin(np.where(np.isnan(ts[:, nosched]), np.inf, np.abs(block[:, None] - ts[:, nosched])), axis=1)
    excl = d10[:, sched_i]
    d_other = np.where(excl, np.inf, dmin_other)
    a1_union = 100 * copy_mass(np.where(np.isfinite(d_other), d_other, np.nan), len(block))
    a1_sum = float(sum(v[0] for k, v in cm.items() if k != "SCHED"))
    a1_sum_corr = float(sum(v[1] for k, v in cm.items() if k != "SCHED"))
    a2 = oracle_a2(af)
    gate = {"A1_copy_mass_union_non_sched_pct": a1_union, "A1_copy_mass_sum_non_sched_pct": a1_sum,
            "A1_copy_mass_sum_non_sched_corrected_pct": a1_sum_corr, "A1_threshold_pct": 5.0,
            "A1_pass": bool(max(a1_union, a1_sum_corr) >= 5.0),
            "A2_oracle": a2, "A2_threshold_delta_s": -15.0, "A2_pass": bool(a2["weighted"]["delta"] <= -15.0),
            "row_summary": row_summary, "aux_stand": aux_stats}
    gate["gate_A_pass"] = gate["A1_pass"] and gate["A2_pass"]
    (OUT / "gate_a.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20)
    print(cov.select("anchor", "pct_available", *[f"pct_within_{t}s" for t in TOLS], "copy_mass_pct", "copy_mass30_pct", "copy_mass_corrected_pct", "median_d_within_120"))
    print(json.dumps({k: v for k, v in gate.items() if k not in ("row_summary",)}, indent=1, default=float))
    print(json.dumps(row_summary, indent=1, default=float))


if __name__ == "__main__":
    main()
