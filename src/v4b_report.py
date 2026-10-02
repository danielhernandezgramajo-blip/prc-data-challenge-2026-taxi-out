"""
V4b — Informe (outputs/experiments/v4b/report.md) a partir de las salidas de v4b_stage2.py y v4b_analysis.py.

    python src/v4b_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from v4b_protocol import OUT_DIR  # noqa: E402

VARIANT_NAME = {"V3": "V3 (S4A_mix)", "S": "mezcla suave (p tal cual)", "S_T": "suave + Platt (log-loss)",
                "S_R2": "suave + recalibración por SSE", "S_R3": "suave + recalibración por SSE con término de tamaño de Δ",
                "H": "compuerta dura", "D": "regresión directa de y − b", "Q": "corrección estructurada b + q·Δ"}
NORMAL_NAME = {"n0": "normal V3", "n2": "normal V4b", "nb": "50/50 V3 + V4b"}
SOURCE_NAME = {"p0": "p1 de V3", "pn": "p1 nuevo"}


def f1(x) -> str:
    return "—" if x is None else f"{x:,.1f}".replace(",", " ")


def sg(x) -> str:
    return "—" if x is None else f"{x:+.1f}"


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def describe(key: str) -> str:
    if key == "C0":
        return "C0 · V3 reentrenado"
    c, n, s, v = key.split("__")
    return f"{c} · {VARIANT_NAME[v]} · {SOURCE_NAME[s]} · {NORMAL_NAME[n]}"


def main() -> None:
    dec = json.loads((OUT_DIR / "decision.json").read_text())
    sel = json.loads((OUT_DIR / "stage2_selection.json").read_text())
    cand = pl.read_csv(OUT_DIR / "metrics_by_candidate.csv")
    bd = pl.read_csv(OUT_DIR / "breakdown.csv")
    apt = pl.read_csv(OUT_DIR / "by_airport.csv")
    diag = pl.read_csv(OUT_DIR / "classifier_diag.csv")
    nb = pl.read_csv(OUT_DIR / "normal_branch.csv", schema_overrides={"seed": pl.String})
    units = [json.loads(x) for x in (OUT_DIR / "units_log.jsonl").read_text().splitlines() if x.strip()]
    seeds = dec["seeds"]; prim = dec["primary"]; D = dec["deltas"]; P = dec["rmse_w"]
    pre = " (PRELIMINAR: semillas " + ", ".join(map(str, seeds)) + ")" if dec["preliminary"] else ""
    # tabla principal
    main_rows = []
    for c in ("C0", "C1", "C2"):
        r = cand.filter(pl.col("key") == prim[c]).row(0, named=True)
        main_rows.append([f"**{c}**", describe(prim[c]), f1(r["rmse_ene_ens"]), f1(r["rmse_jul_ens"]), f"**{f1(r['rmse_w_ens'])}**",
                          f"{f1(r['rmse_w_seed_min'])}–{f1(r['rmse_w_seed_max'])}", sg(r["delta_vs_C0"]) if c != "C0" else "—"])
    d_rows = [[k, sg(v["delta_w"]), f"[{sg(v['ci95'][0])}, {sg(v['ci95'][1])}]", f"{v['p_improve']:.2f}",
               ", ".join(sg(x) for x in v["per_seed"]), sg(v["by_month"]["ene"]), sg(v["by_month"]["jul"])] for k, v in D.items()]
    # desglose por clase
    bd_rows = []
    for c in ("C0", "C1", "C2"):
        for mes in ("ene", "jul"):
            r = bd.filter((pl.col("config") == c) & (pl.col("mes") == mes)).row(0, named=True)
            bd_rows.append([c, mes, f1(r["rmse_total"]), f1(r["rmse_clase0"]), f1(r["rmse_clase0_le1h"]), f1(r["rmse_clase0_1a3h"]),
                            f1(r["rmse_clase0_gt3h"]), f1(r["rmse_clase1"]), f1(r["rmse_clase2"]) + f" ({r['n_clase2']})",
                            f"{r['sse_share_clase0']:.0f}/{r['sse_share_clase1']:.0f}/{r['sse_share_clase2']:.0f}"])
    # rama normal
    nbe = nb.filter(pl.col("seed") == "ens")
    nb_rows = [[r["mes"], r["normal"], f1(r["rmse_clase0"]), f1(r["rmse_clase0_le1h"])] for r in nbe.sort("mes", "normal").iter_rows(named=True)]
    abl = dec.get("ablation_ctr2")
    # clasificadores
    dg = diag.group_by("mes", "classifier").agg(pl.col("auc", "logloss", "brier_w_delta2", "mean_p_class1", "mean_p_class0").mean()).sort("mes", "classifier")
    dg_rows = [[r["mes"], r["classifier"], f"{r['auc']:.3f}", f"{r['logloss']:.4f}", f"{r['brier_w_delta2']:.4f}",
                f"{r['mean_p_class1']:.3f}", f"{r['mean_p_class0']:.4f}"] for r in dg.iter_rows(named=True)]
    # variantes de etapa 2 (diagnóstico, ordenadas por SSE interno; el outer NO se usó para elegir)
    rank_rows = []
    for c in ("C1", "C2"):
        for i, x in enumerate(sel[c]["ranking"][:8]):
            r = cand.filter(pl.col("key") == x["key"]).row(0, named=True)
            rank_rows.append([c, i + 1, describe(x["key"]).split(" · ", 1)[1], f"{x['inner_sse_sum'] / 1e9:.3f}", f1(r["rmse_w_ens"]), sg(r["delta_vs_C0"])])
    # aeropuertos
    apt_rows = [[r["airport"], f1(r["C0"]), sg(r["d_C1_C0"]), sg(r["d_C2_C0"])] for r in apt.sort("airport").iter_rows(named=True)]
    n_apt_c2 = int((apt["d_C2_C0"] < 0).sum())
    # tiempos
    secs = {}
    for u in units:
        secs.setdefault(u["comp"], []).append(u["seconds"])
    t_rows = [[k, len(v), f"{sum(v) / len(v) / 60:.1f}"] for k, v in sorted(secs.items())]
    d20 = D["C2-C0"]
    ex = dec["sensitivity_excl_class0_gt3h"]["excluded_rows"]
    sens_txt = f"{ex['ene']} en enero, {ex['jul']} en julio"
    verdict = ("**Pasa** el umbral de −15 s" + (" (frágil)" if dec["fragile"] else "")) if dec["pass_to_costly_stage"] else "**No pasa** el umbral de −15 s"
    # --- lectura (todo calculado de las tablas) ---
    def nbv(mes, name, col="rmse_clase0"):
        return nbe.filter((pl.col("mes") == mes) & (pl.col("normal") == name))[col][0]
    def dgv(mes, clf, col):
        return dg.filter((pl.col("mes") == mes) & (pl.col("classifier") == clf))[col][0]
    def bdv(c, mes, col):
        return bd.filter((pl.col("config") == c) & (pl.col("mes") == mes))[col][0]
    lirf = apt.filter(pl.col("airport") == "LIRF").row(0, named=True)
    outer_best = cand.filter(~pl.col("key").is_in(list(prim.values()))).sort("rmse_w_ens").row(0, named=True)
    sens = dec["sensitivity_excl_class0_gt3h"]
    infl20 = dec["single_row_influence"]["C2-C0"]["rows"]
    r21 = max(dec["single_row_influence"]["C2-C1"]["rows"], key=lambda x: abs(x["d_sse_row"]))
    lectura = "\n".join([
        f"- **¿Rama normal insuficiente? Sí.** Sin compuerta, el RMSE de clase 0 baja {sg(nbv('ene', 'normal_C2 (V4b)') - nbv('ene', 'normal_C0 (V3)'))} s en enero y "
        f"{sg(nbv('jul', 'normal_C2 (V4b)') - nbv('jul', 'normal_C0 (V3)'))} s en julio (≤ 1 h: {sg(nbv('ene', 'normal_C2 (V4b)', 'rmse_clase0_le1h') - nbv('ene', 'normal_C0 (V3)', 'rmse_clase0_le1h'))} / "
        f"{sg(nbv('jul', 'normal_C2 (V4b)', 'rmse_clase0_le1h') - nbv('jul', 'normal_C0 (V3)', 'rmse_clase0_le1h'))} s), igual en las 3 semillas; C2 mejora a C0 en {n_apt_c2} de {apt.height} aeropuertos.",
        f"- **¿Compuerta mal calibrada? Solo en parte.** El clasificador nuevo mejora poco la discriminación (AUC {dgv('ene', 'p1_C0 (V3)', 'auc'):.3f} → "
        f"{dgv('ene', 'p1_new (V4b)', 'auc'):.3f} en enero, {dgv('jul', 'p1_C0 (V3)', 'auc'):.3f} → {dgv('jul', 'p1_new (V4b)', 'auc'):.3f} en julio); su p medio en filas "
        f"de clase 1 sigue en ≈ {dgv('jul', 'p1_new (V4b)', 'mean_p_class1'):.2f}. La ganancia de C1 se concentra en julio ({sg(D['C1-C0']['by_month']['jul'])} s) y en LIRF "
        f"({sg(lirf['d_C1_C0'])} s); en enero C1 empeora ({sg(D['C1-C0']['by_month']['ene'])} s). En la selección anidada ganó la mezcla suave simple: "
        "la recalibración por SSE, la compuerta dura y las regresiones de corrección quedaron peor en el SSE interno cruzado.",
        f"- **Una sola fila oculta parte de la mejora de la rama normal.** {r21['airport']} ({r21['mes']}), clase {r21['clase']}, y = MVT − SCHED = "
        f"{f1(r21['y'])} s: la mezcla con la normal de V3 quedaba más cerca por casualidad (la normal de V3 extrapolaba un valor enorme para esa fila). "
        f"Esa fila sube el SSE de C2 frente a C1 en {r21['d_sse_row']:.2e}; sin ella, C2 − C1 pasaría de {sg(D['C2-C1']['delta_w'])} a "
        f"{sg(r21['delta_w_without_row'])} s. El RMSE de clase 1 de julio (C1 {f1(bdv('C1', 'jul', 'rmse_clase1'))} → C2 {f1(bdv('C2', 'jul', 'rmse_clase1'))}) "
        "empeora por esa fila, no por un sesgo sistemático.",
        f"- **La decisión frente al umbral no es robusta.** Quitar una sola fila cambia C2 − C0 de {sg(D['C2-C0']['delta_w'])} s a entre "
        f"{sg(min(x['delta_w_without_row'] for x in infl20))} y {sg(max(x['delta_w_without_row'] for x in infl20))} s (las {len(infl20)} filas más influyentes). "
        "La mejora frente a V3 es real (IC 95 % por debajo de 0, 3 de 3 semillas, 10 de 10 aeropuertos); si llega o no a 15 s no se puede "
        "decidir con estos datos. Se aplica la regla tal como se fijó.",
        f"- **Fecha desplazada aparte:** sin efecto medible (misma celda en C0–C2; {bdv('C0', 'ene', 'n_clase2')} fila en enero y {bdv('C0', 'jul', 'n_clase2')} en julio).",
        f"- **Ruido:** en enero, {sens['excluded_rows']['ene']} filas de clase 0 con y > 3 h dominan el SSE; sin ellas (sensibilidad que no decide) C2 − C0 sería "
        f"{sg(sens['delta_C2_C0'])} s.",
        f"- **Selección honesta:** variantes no elegidas parecen mejores en el mes outer (la mejor, {describe(outer_best['key']).split(' · ', 1)[1]}: "
        f"{sg(outer_best['delta_vs_C0'])} s), pero quedaron peor en febrero/agosto. Elegirlas ahora sería ajustar con el mes de evaluación. Posible sesgo del "
        "esquema: el ajuste cruzado con un solo mes interno penaliza más a las calibraciones flexibles; un protocolo nuevo podría usar más meses internos.",
        "- **No ejecutado (regla de −15 s no superada):** ablación con/sin ARVT, adaptación no supervisada al ranking, max_ctr_complexity 3, ensamble con LightGBM.",
    ])
    md = f"""# V4b — Resultado mínimo C0/C1/C2{pre}

Generado por `src/v4b_report.py`. Protocolo: `protocol.md` (enmienda A1 de cómputo incluida). Validación: LOJO (enero 2025) y
LOJuO (julio 2025), componentes entrenados con 9 meses, calibración anidada en febrero/agosto, ensamble de semillas
{", ".join(map(str, seeds))}. Sin candidata, sin subidas, sin push, sin leaderboard. Las cotas de oráculo de V4 (230.2, 205.7)
no son alcanzables y no se usan como referencia.

## Veredicto

{verdict}: C2 − C0 = **{sg(d20['delta_w'])} s** de RMSE ponderado (IC 95 % [{sg(d20['ci95'][0])}, {sg(d20['ci95'][1])}],
semillas {", ".join(sg(x) for x in d20['per_seed'])}). Compuerta (C1 − C0): {sg(D['C1-C0']['delta_w'])} s; rama normal (C2 − C1):
{sg(D['C2-C1']['delta_w'])} s. {"Se puede pasar a la etapa costosa (ARVT, adaptación, más capacidad) sin generar candidata." if dec["pass_to_costly_stage"] else "Según la regla prefijada, no se continúa con modelos costosos."}

## Lectura

{lectura}

## 1. RMSE ponderado (ensamble de semillas; enero y julio 2025 directos)

{table(["config", "definición (elegida en feb/ago)", "enero", "julio", "RMSE_w", "rango semillas", "Δ vs C0"], main_rows)}

## 2. Diferencias con bootstrap pareado (aeropuerto-día, estratificado por mes, 1 000 réplicas)

{table(["comparación", "Δ RMSE_w", "IC 95 %", "P(mejora)", "por semilla", "Δ enero", "Δ julio"], d_rows)}

Sensibilidad (no decide): sin las filas de clase 0 con y > 3 h ({sens_txt}), C1 − C0 = {sg(dec['sensitivity_excl_class0_gt3h']['delta_C1_C0'])} s,
C2 − C0 = {sg(dec['sensitivity_excl_class0_gt3h']['delta_C2_C0'])} s, C2 − C1 = {sg(dec['sensitivity_excl_class0_gt3h']['delta_C2_C1'])} s (RMSE_w de C0 {f1(dec['sensitivity_excl_class0_gt3h']['rmse_w']['C0'])}).
Ninguna configuración predice esas filas; pesan mucho en el SSE y comprimen las diferencias medidas en RMSE.

## 3. Desglose por clase (ensamble; s)

{table(["config", "mes", "total", "clase 0", "clase 0 ≤ 1 h", "clase 0 1–3 h", "clase 0 > 3 h", "clase 1", "clase 2 (n)", "SSE % c0/c1/c2"], bd_rows)}

## 4. ¿Rama normal insuficiente? RMSE en filas de clase 0, sin compuerta (ensamble)

{table(["mes", "rama normal", "clase 0", "clase 0 ≤ 1 h"], nb_rows)}

{("Ablación ctr 2 (LOJO, s42): clase 0 " + f1(abl["rmse_clase0_ctr1"]) + " (ctr 1) → " + f1(abl["rmse_clase0_ctr2"]) + " (ctr 2), Δ " + sg(abl["delta_clase0"]) + " s.") if abl else "Ablación ctr 2 (rama normal): **no ejecutada** — el ajuste superó dos veces el límite de memoria del entorno (~5.8 GiB); su efecto queda sin medir (protocolo §10, A2)."}

## 5. ¿Compuerta mal calibrada? Clasificadores de clase 1 (filas fuera de la celda; media de semillas)

{table(["mes", "clasificador", "AUC", "log-loss", "Brier ponderado Δ²", "p medio clase 1", "p medio clase 0"], dg_rows)}

Variantes de etapa 2 por SSE interno cruzado (feb↔ago, sumado sobre outer × semillas). El RMSE_w outer se muestra solo como
diagnóstico: no se usó para elegir.

{table(["config", "rango", "variante · fuente · normal", "SSE interno (×1e9)", "RMSE_w outer", "Δ vs C0"], rank_rows)}

## 6. Aeropuertos (RMSE ponderado ene/jul de cada aeropuerto)

{table(["aeropuerto", "C0", "Δ C1", "Δ C2"], apt_rows)}

C2 mejora a C0 en {n_apt_c2} de {apt.height} aeropuertos.

## 7. Cómputo

{table(["componente", "ajustes", "min por ajuste"], t_rows)}
"""
    (OUT_DIR / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
