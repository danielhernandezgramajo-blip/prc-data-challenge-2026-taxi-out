"""
V4 — Informes de la Etapa A y decisión, generados a partir de las salidas de v4_audit.py y v4_oracles.py.

    python src/v4_report.py

Salidas (outputs/experiments/v4/): source_audit.md, report.md, decision.json. Verifica en modo lectura los hashes de los
archivos protegidos presentes en este entorno. No entrena, no sube nada y no usa el leaderboard.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments" / "v4"
BASE_HEAD = "056bbc4"
PROTECTED_SHA256 = {
    "submissions/candidate_v1.parquet": "d482a42b3230d7e9eeba3890a66fc5ab4b9362d5e54a33c0151a265cfb5c4294",
    "submissions/candidate_v2.parquet": "e1db051a419e18d7314b56fb5b8105c93170f99c4e4254b1aeab7f8fc7f84752",
    "submissions/candidate_v3.parquet": "165c6e298a033d2ad96765e3ff3652615f20d6411a15ad6acd144b76ef920ee0",
    "submissions/candidate_v1.json": "8b6a4aec50730d61f9d96ef5bd2657b0a342fcfea79ab1684bcd3795c796f39d",
    "submissions/candidate_v2.json": "6cb176329a9cbb92ea9e0193567b8663663f2831b03224d303d9c8a9e5290cfc",
    "submissions/candidate_v3.json": "cde6ea191546ff21d9636d50bb341f4c5efea79eb5d9958b55a6fb834ba20b35",
    "outputs/_mac_tracked_md5.txt": "301e9aea5ec4ccc42828af374b008e2730db5b788f88eea16f7cfd97b16bc672",
}
PARQUET_JOINT_SHA256 = "1df017af055bdbb77d69da4ac2833b7134380761e04ab811172e5f9292b927ac"   # `sha256sum *.parquet | sha256sum`
BUNDLES_JOINT_MD5 = "28fd7b74aa08ae34bfccc1cc2b66c15e"        # `find <BUNDLE_DIRS> -type f | sort | xargs md5sum | md5sum`
BUNDLE_DIRS = ("models/final", "models/experiments/v2/final", "models/experiments/v3/final")
BASE = ["SCHED", "IOBT", "LOBT", "EOBT1", "AOBT3"]
NAME = {"SCHED": "SCHED_TIME", "IOBT": "IOBT", "LOBT": "LOBT", "EOBT1": "EOBT_1", "AOBT3": "AOBT_3"}


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_joint_sha256() -> str | None:
    pq = sorted(p.name for p in ROOT.glob("*.parquet"))
    if len(pq) != 14:
        return None
    return hashlib.sha256("".join(f"{digest(ROOT / n, 'sha256')}  {n}\n" for n in pq).encode()).hexdigest()


def bundles_joint_md5() -> str | None:
    files = sorted(str(p.relative_to(ROOT)) for d in BUNDLE_DIRS for p in (ROOT / d).rglob("*") if p.is_file())
    if len(files) != 43:
        return None
    return hashlib.md5("".join(f"{digest(ROOT / f, 'md5')}  {f}\n" for f in files).encode()).hexdigest()


def verify_protected() -> dict:
    """Solo lectura. En el espejo de cómputo (no git) los metadatos .json son borradores previos: solo se comprueban .parquet."""
    authoritative = (ROOT / ".git").exists()
    files = {}
    for rel, sha in PROTECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists() or (not authoritative and not rel.endswith(".parquet")):
            files[rel] = "no comprobado en este entorno"
        else:
            files[rel] = "ok" if digest(p, "sha256") == sha else "DISTINTO"
    pj, bj = parquet_joint_sha256(), bundles_joint_md5()
    files["14 Parquet del challenge (sha256 conjunto)"] = "no comprobado en este entorno" if pj is None else ("ok" if pj == PARQUET_JOINT_SHA256 else "DISTINTO")
    files["43 archivos de bundles finales (md5 conjunto)"] = "no comprobado en este entorno" if bj is None else ("ok" if bj == BUNDLES_JOINT_MD5 else "DISTINTO")
    return {"environment": "copia de trabajo git (autoritativa)" if authoritative else "espejo de cómputo en la nube (no git)",
            "files": files, "all_checked_ok": all(v != "DISTINTO" for v in files.values()),
            "n_checked": sum(v == "ok" for v in files.values())}


def nf(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def f1(x: float) -> str:
    return f"{x:,.1f}".replace(",", " ")


def f2(x: float) -> str:
    return f"{x:.2f}"


def _auc(v) -> str:
    return "n/a" if v is None or v != v else f"{v:.3f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load() -> dict:
    d = {"gate": json.loads((OUT / "gate_a.json").read_text()), "orc": json.loads((OUT / "oracles.json").read_text()),
         "cov": pl.read_csv(OUT / "source_coverage.csv"), "apt": pl.read_csv(OUT / "source_attribution_by_airport.csv"),
         "mon": pl.read_csv(OUT / "source_attribution_by_month.csv"), "conf": pl.read_csv(OUT / "source_confusion.csv"),
         "gi": pl.read_csv(OUT / "source_group_informativeness.csv"), "res": pl.read_csv(OUT / "timestamp_resolution.csv"),
         "aux": pl.read_csv(OUT / "aux_stand_anchors.csv"), "oapt": pl.read_csv(OUT / "oracle_by_airport.csv")}
    return d


def source_audit_md(d: dict) -> str:
    g, o, cov = d["gate"], d["orc"]["steps"], d["cov"]
    rs = g["row_summary"]; a2 = g["A2_oracle"]
    cv = {r["anchor"]: r for r in cov.iter_rows(named=True)}
    shifted = [a for a in cv if a.endswith("d")]
    max_shift = max(cv[a]["pct_within_300s"] for a in shifted)
    cov_rows = []
    for a in BASE + ["UNION_all", "UNION_all_but_SCHED"]:
        r = cv[a]
        cov_rows.append([NAME.get(a, a.replace("UNION_all_but_SCHED", "unión sin SCHED").replace("UNION_all", "unión (15 anclas)")),
                         f2(r["pct_available"]), f2(r["pct_within_10s"]), f2(r["pct_within_30s"]), f2(r["pct_within_60s"]),
                         f2(r["pct_within_300s"]),
                         "—" if r["copy_mass_pct"] is None else f2(r["copy_mass_pct"]),
                         "—" if r["copy_mass_corrected_pct"] is None else f2(r["copy_mass_corrected_pct"]),
                         "—" if r["median_d_within_120"] is None else f"{r['median_d_within_120']:+.0f}"])
    near = rs["nearest_anchor_share_pct"]
    conf = d["conf"].filter(pl.col("anchor_A").is_in(BASE) & pl.col("anchor_B").is_in(BASE))
    conf_rows = []
    for a in BASE:
        sub = conf.filter(pl.col("anchor_A") == a)
        n = sub["n_A_within"][0]
        conf_rows.append([NAME[a], f"{n:,}".replace(",", " ")] + [f1(sub.filter(pl.col("anchor_B") == b)["pct_B_within_given_A"][0]) for b in BASE])
    apt = d["apt"]
    apt_rows = [[r["ADEP_mvt"], f"{r['n']:,}".replace(",", " ")] + [f2(r.get(c, 0.0) or 0.0) for c in ("SCHED", "IOBT", "LOBT", "EOBT1", "AOBT3", "none")]
                for r in apt.iter_rows(named=True)]
    mon = d["mon"]
    rng = {c: (mon[c].min(), mon[c].max()) for c in ("SCHED", "IOBT", "EOBT1", "AOBT3", "none")}
    gi = d["gi"]
    gi_rows = []
    for grp in ["airport", "operator", "flight_number", "callsign", "airport_stand", "airport_runway", "route", "presence_pattern", "aobt3_resolution", "month"]:
        sub = gi.filter(pl.col("grouping") == grp)
        gi_rows.append([grp, f"{sub['n_groups'][0]:,}".replace(",", " ")] +
                       [_auc(sub.filter(pl.col('target') == t)['auc_out_of_month'][0]) for t in ("copy_SCHED", "copy_AOBT3", "copy_IOBT_LOBT_EOBT_only", "copy_any")])
    res = d["res"]
    sec0 = lambda c: (res[f"{c}__pct_sec0"].min(), res[f"{c}__pct_sec0"].max())
    eham_aobt = res.filter(pl.col("ADEP_mvt") == "EHAM")["AOBT_3_flt__pct_sec0"][0]
    aux = g["aux_stand"]
    rm = {k: v["rows_modified"] for k, v in o.items()}
    a2t = o["A2_nonSCHED_tau10"]
    lebl = d["oapt"].filter(pl.col("airport") == "LEBL")["delta__A2_nonSCHED_tau10"][0]
    n_apt15 = d["orc"]["airports_A2_tau10_delta_le_-15s"]
    best_auc = gi.filter((pl.col("grouping") != "month") & pl.col("auc_out_of_month").is_not_nan()).sort("auc_out_of_month", descending=True).row(0, named=True)
    apt_auc = gi.filter((pl.col("grouping") == "airport") & (pl.col("target") == best_auc["target"]))["auc_out_of_month"][0]
    max_aobt_auc = gi.filter((pl.col("target") == "copy_AOBT3") & (pl.col("grouping") != "month"))["auc_out_of_month"].max()

    return f"""# V4 · Etapa A — Auditoría de las fuentes visibles de BLOCK_TIME

Generado por `src/v4_report.py` a partir de las salidas de `src/v4_audit.py` (CSV/JSON de esta carpeta) y `src/v4_oracles.py`.
Datos: {nf(g['A2_oracle']['Jul']['n'])} salidas de julio y {nf(g['A2_oracle']['Dec']['n'])} de diciembre para los oráculos; 2 085 047 salidas (DEP)
de training 2025 para la auditoría. Protocolo prerregistrado: `protocol.md` (guardado antes de esta auditoría).

**Resultado: Gate A NO superado.** A1 pasa (masa de copia no-SCHED {f2(g['A1_copy_mass_union_non_sched_pct'])} % ≥ 5 %), A2 falla
(oráculo {a2['weighted']['delta']:+.2f} s, IC 95 % [{a2t['ci95_delta_w'][0]:+.1f}, {a2t['ci95_delta_w'][1]:+.1f}] s, frente a −15 s exigidos).

## 1. Definiciones

- **Ancla**: timestamp visible en ranking del que BLOCK podría haberse copiado: SCHED_TIME, IOBT, LOBT, EOBT_1 y AOBT_3, cada
  uno con desplazamiento 0, −1 día y +1 día (15 anclas). `d = BLOCK − ancla`.
- **Atribución por fila**: ancla más cercana si |d| ≤ τ (τ = 30 s prefijado; 10 s = copia estricta). Empates exactos por
  precedencia SCHED > IOBT > LOBT > EOBT_1 > AOBT_3 (no altera la predicción: las anclas coinciden). Ambigua = ≥ 2 anclas
  dentro de τ con timestamps que difieren > 10 s.
- **Masa de copia**: exceso de filas con |d| ≤ 10 s sobre el fondo local (densidad por segundo en 30 < |d| ≤ 120 s), en % de
  todas las DEP. Separa copias de simples cercanías entre dos medidas del mismo evento. *Corregida*: d centrado con la moda por
  (aeropuerto, ancla) — la "corrección residual" de la hipótesis.

## 2. Cobertura por fuente (% de todas las DEP 2025)

{table(["fuente", "disponible", "±10 s", "±30 s", "±60 s", "±300 s", "masa de copia", "corregida", "mediana d (|d|≤120 s)"], cov_rows)}

- Las 10 anclas desplazadas ±1 día no explican nada: la mayor cubre {max_shift:.4f} % de las DEP a ±300 s. La clase de fecha
  desplazada de V3 (y ≈ 86 400 s + rodaje, 13 filas en 2025) no es una copia de una fuente visible desplazada; es BLOCK con la
  fecha equivocada (experto `none − 86 400` del protocolo).
- La "corrección residual" apenas cambia nada: la suma de masas no-SCHED pasa de {f2(g['A1_copy_mass_sum_non_sched_pct'])} % a
  {f2(g['A1_copy_mass_sum_non_sched_corrected_pct'])} % (la mediana de d entre las filas con |d| ≤ 120 s está a +1/+3 s según la fuente).
- La suma de masas por ancla cuenta varias veces la misma fila (IOBT/LOBT/EOBT_1 suelen coincidir con SCHED). Sin doble
  conteo, la masa de copia de las anclas no-SCHED en filas que no son copia de SCHED es **{f2(g['A1_copy_mass_union_non_sched_pct'])} %**.

## 3. Atribución por fila

- τ = 30 s: **{f2(rs['pct_rows_attributed_tau30'])} %** de las DEP se atribuyen a alguna ancla; τ = 10 s: {f2(rs['pct_rows_attributed_tau10'])} %.
  Ambiguas: {rs['pct_ambiguous_tau30']:.3f} %. Atribuidas a anclas desplazadas: {rs['pct_attributed_to_shifted_anchor_tau30']:.5f} %.
- Ancla más cercana (sin tolerancia): AOBT_3 {f1(near['AOBT3'])} %, SCHED {f1(near['SCHED'])} %, EOBT_1 {f1(near['EOBT1'])} %,
  IOBT {f1(near['IOBT'])} %, LOBT {near['LOBT']:.2f} % (LOBT casi siempre coincide con IOBT y pierde por precedencia).
  Distancia a la más cercana: p25 {rs['dist1_quantiles_s'][1]:.0f} s, mediana {rs['dist1_quantiles_s'][2]:.0f} s, p75 {rs['dist1_quantiles_s'][3]:.0f} s.
- Fuentes ausentes: IOBT, LOBT, EOBT_1 y AOBT_3 faltan juntas en el {rs['pct_rows_missing_source']['AOBT_3_flt']:.2f} % de las DEP
  (sin datos `_flt`); SCHED nunca falta.

## 4. Solapamiento entre fuentes (de las filas con BLOCK a ±30 s de la fuente de la fila, % que también está a ±30 s de la de la columna)

{table(["si BLOCK ≈", "n filas", "SCHED", "IOBT", "LOBT", "EOBT_1", "AOBT_3"], conf_rows)}

Las copias de IOBT/LOBT/EOBT_1 son mayoritariamente la misma fila que la copia de SCHED (los planes coinciden con el horario).
AOBT_3 es la única fuente realmente distinta (solapa un 5–7 % con las de plan).

## 5. Por aeropuerto y por mes (atribución a τ = 30 s, % de DEP)

{table(["aeropuerto", "DEP", "SCHED", "IOBT", "LOBT", "EOBT_1", "AOBT_3", "ninguna"], apt_rows)}

Por mes la atribución es estable (rango ene–dic): SCHED {f2(rng['SCHED'][0])}–{f2(rng['SCHED'][1])} %, IOBT {f2(rng['IOBT'][0])}–{f2(rng['IOBT'][1])} %,
EOBT_1 {f2(rng['EOBT1'][0])}–{f2(rng['EOBT1'][1])} %, AOBT_3 {f2(rng['AOBT3'][0])}–{f2(rng['AOBT3'][1])} %, ninguna {f2(rng['none'][0])}–{f2(rng['none'][1])} %.

## 6. Gate A (prefijado en `protocol.md` §2)

| condición | valor | umbral | resultado |
|---|---|---|---|
| A1 · masa de copia no-SCHED (unión sin doble conteo; suma corregida) | {f2(g['A1_copy_mass_union_non_sched_pct'])} %; {f2(g['A1_copy_mass_sum_non_sched_corrected_pct'])} % | ≥ 5 % | {'pasa' if g['A1_pass'] else 'falla'} |
| A2 · oráculo no-SCHED ±10 s sobre S4A_mix (V3), RMSE ponderado | {f2(a2['weighted']['rmse_s4a'])} → {f2(a2['weighted']['rmse_oracle'])} s (Δ {a2['weighted']['delta']:+.2f} s) | Δ ≤ −15 s | {'pasa' if g['A2_pass'] else '**falla**'} |

A2 por mes: julio {f1(a2['Jul']['rmse_s4a'])} → {f1(a2['Jul']['rmse_oracle'])} s ({nf(a2['Jul']['n_oracle_rows'])} filas, {f2(a2['Jul']['pct_oracle_rows'])} %);
diciembre {f1(a2['Dec']['rmse_s4a'])} → {f1(a2['Dec']['rmse_oracle'])} s ({nf(a2['Dec']['n_oracle_rows'])} filas, {f2(a2['Dec']['pct_oracle_rows'])} %).
Bootstrap pareado (1 000 réplicas, bloques aeropuerto-día, estratificado por mes): Δ IC 95 % [{a2t['ci95_delta_w'][0]:+.1f}, {a2t['ci95_delta_w'][1]:+.1f}] s.
**Gate A: {'superado' if g['gate_A_pass'] else 'NO superado → la etapa termina aquí (protocolo §2).'}**

## 7. Por qué la hipótesis no es viable

1. **Las copias no-SCHED existen, pero caen donde V3 ya acierta.** Las filas que el oráculo corrige ({f2(rm['A2_nonSCHED_tau10']['Jul']['pct'])} % en julio,
   {f2(rm['A2_nonSCHED_tau10']['Dec']['pct'])} % en diciembre) tienen un RMSE de V3 de {f1(rm['A2_nonSCHED_tau10']['Jul']['v3_rmse_on_rows'])} / {f1(rm['A2_nonSCHED_tau10']['Dec']['v3_rmse_on_rows'])} s —por debajo de la media— y
   concentran solo el {f2(rm['A2_nonSCHED_tau10']['Jul']['v3_sse_share_pct'])} % / {f2(rm['A2_nonSCHED_tau10']['Dec']['v3_sse_share_pct'])} % del error cuadrático. Son mayoritariamente filas con BLOCK ≈ AOBT_3, y
   `MVT − AOBT_3` ya es una variable de V3. Aun sabiendo con certeza cuáles son, el RMSE baja ~5 s.
2. **El RMSE lo dominan filas que ninguna fuente visible explica.** En julio, 4 filas de fecha desplazada aportan el
   {f2(rm['C_dayshift']['Jul']['v3_sse_share_pct'])} % del SSE de V3; la cola > 1 h que no es copia ({rm['E_perfect_tail']['Jul']['n']} filas, {f2(rm['E_perfect_tail']['Jul']['pct'])} %) aporta el
   {f2(rm['E_perfect_tail']['Jul']['v3_sse_share_pct'])} %; las filas que copian alguna fuente ({f2(rm['B_all_anchors']['Jul']['pct'])} %) concentran el
   {f2(rm['B_all_anchors']['Jul']['v3_sse_share_pct'])} %, casi todo por las copias de SCHED (que V3 ya modela con p1), no por las no-SCHED.
   Ni siquiera una compuerta perfecta sobre todas las fuentes, SCHED incluida, baja de 260 s: {f1(o['B_all_anchors']['rmse_w'])} s.
3. **Robustez.** Con τ = 30 s el oráculo gana {o['A2_nonSCHED_tau30']['delta_w_vs_v3']:+.1f} s; con τ = 60 s, {o['A2_nonSCHED_tau60']['delta_w_vs_v3']:+.1f} s (y a ±60 s ya no
   son copias: el oráculo "acierta" el {f1(rm['A2_nonSCHED_tau60']['Jul']['pct'])} % de las filas). Por aeropuerto solo {n_apt15} de 10 llega a −15 s
   (LEBL, {lebl:+.1f} s). Ambas aproximaciones (masa sobre fondo local, oráculo a ±10 s) favorecen a la hipótesis; aun así no alcanza.

## 8. ¿Hay señal visible para identificar qué filas son copia? (AUC con tasa del grupo estimada fuera de mes)

{table(["agrupación (visible en ranking)", "grupos", "copia SCHED", "copia AOBT_3", "copia solo IOBT/LOBT/EOBT_1", "cualquier copia"], gi_rows)}

La mejor agrupación (`{best_auc['grouping']}` para `{best_auc['target']}`) llega a {best_auc['auc_out_of_month']:.3f}, solo
{best_auc['auc_out_of_month'] - apt_auc:+.3f} sobre el aeropuerto por sí solo ({apt_auc:.3f}). Para las copias de AOBT_3 ninguna supera
{max_aobt_auc:.3f}. El patrón de presencia no informa (≈ 0.5; solo hay 2 patrones: con o sin datos `_flt`). Nota metodológica: una primera versión usaba la tasa
leave-one-out por fila, que en agrupaciones de baja cardinalidad con una categoría dominante sesga el AUC muy por debajo de 0.5
(p. ej. 0.01 para el patrón de presencia); se sustituyó por la estimación fuera de mes, con el prior del suavizado calculado
también sin el mes evaluado (corrección posterior: la versión del commit c68179e usaba un prior global, una fuga mínima).
Estos AUC son univariados y no acotan un clasificador multivariable. `month` no tiene AUC: fuera de mes no generaliza.

## 9. Resolución de los timestamps (% con segundos = 0, rango entre aeropuertos)

BLOCK {f1(sec0('BLOCK_TIME_UTC_mvt')[0])}–{f1(sec0('BLOCK_TIME_UTC_mvt')[1])} %, MVT {f1(sec0('MVT_TIME_UTC_mvt')[0])}–{f1(sec0('MVT_TIME_UTC_mvt')[1])} % (resolución de segundos);
SCHED {f1(sec0('SCHED_TIME_UTC_mvt')[0])}–{f1(sec0('SCHED_TIME_UTC_mvt')[1])} %, IOBT {f1(sec0('IOBT_flt')[0])}–{f1(sec0('IOBT_flt')[1])} %, LOBT {f1(sec0('LOBT_flt')[0])}–{f1(sec0('LOBT_flt')[1])} %,
EOBT_1 {f1(sec0('EOBT_1_flt')[0])}–{f1(sec0('EOBT_1_flt')[1])} %, AOBT_3 {f1(sec0('AOBT_3_flt')[0])}–{f1(sec0('AOBT_3_flt')[1])} % (minuto; EHAM {f1(eham_aobt)} %);
ARVT_1 {f1(sec0('ARVT_1_flt')[0])}–{f1(sec0('ARVT_1_flt')[1])} %, ARVT_3 {f1(sec0('ARVT_3_flt')[0])}–{f1(sec0('ARVT_3_flt')[1])} %. La resolución de AOBT_3 como
agrupación no añade identificación sobre el aeropuerto (tabla §8, `aobt3_resolution`).

## 10. Anclas auxiliares (rotación del stand; no son copias)

El on-block de la llegada siguiente al mismo (aeropuerto, stand), visible en ranking, existe para el {f1(aux['pct_dep_with_next_arr_6h'])} %
de las DEP en 6 h, pero llega una mediana de {aux['next_gap_quantiles_min'][2]:.0f} min después de BLOCK (p5 {aux['next_gap_quantiles_min'][0]:.0f} min) y solo en el
{f2(aux['pct_next_gap_0_10min'])} % de las DEP cae a 0–10 min: cota superior floja, no una fuente de BLOCK.

## 11. Conclusión

Las fuentes visibles explican una fracción real de BLOCK (≈ 15–18 % de las filas a ±10–30 s; ≈ 5.8 % de copias no-SCHED sin
doble conteo), pero esa fracción no es *material para el error*: incluso con una compuerta perfecta la mezcla de fuentes no-SCHED
mejora el RMSE ponderado en ~5 s. Conforme al protocolo, no se ejecutan las Etapas B–E ni se construye candidata.
"""


def report_md(d: dict, ver: dict) -> str:
    g, orc = d["gate"], d["orc"]; o = orc["steps"]
    lab = {"V3_S4A_mix": "C0 de referencia: S4A_mix (V3), OOF origen creciente",
           "A2_nonSCHED_tau10": "Oráculo Gate A: + fuentes no-SCHED perfectas (±10 s)",
           "A2_nonSCHED_tau30": "↳ sensibilidad ±30 s", "A2_nonSCHED_tau60": "↳ sensibilidad ±60 s (ya no son copias)",
           "B_all_anchors": "Compuerta perfecta sobre TODAS las fuentes (incl. SCHED)",
           "C_dayshift": "+ fecha desplazada perfecta", "C2_rest_normal": "+ resto con la rama normal pura (sin masa p1)",
           "D_stage3_bulk": "+ capacidad/escala Etapa 3 en el grueso (aprox.; oráculo, NO alcanzable)", "E_perfect_tail": "+ cola > 1 h perfecta (suelo; NO alcanzable)"}
    rows = []
    for s, lb in lab.items():
        r = o[s]
        rows.append([lb, f1(r["rmse_jul"]), f1(r["rmse_dec"]), f"**{f1(r['rmse_w'])}**",
                     f"[{f1(r['ci95_rmse_w'][0])}, {f1(r['ci95_rmse_w'][1])}]",
                     "—" if s == "V3_S4A_mix" else f"{r['delta_w_vs_v3']:+.1f} [{r['ci95_delta_w'][0]:+.1f}, {r['ci95_delta_w'][1]:+.1f}]"])
    oa = d["oapt"].sort("airport")
    apt_rows = [[r["airport"], f1(r["rmse_w__V3_S4A_mix"]), f"{r['delta__A2_nonSCHED_tau10']:+.1f}", f"{r['delta__B_all_anchors']:+.1f}",
                 f"{r['delta__C2_rest_normal']:+.1f}"] for r in oa.iter_rows(named=True)]
    ver_rows = [[k, v] for k, v in ver["files"].items()]
    a2t = o["A2_nonSCHED_tau10"]
    st = lambda k: o[k]["rmse_w"]
    inc = {"B": st("B_all_anchors") - st("V3_S4A_mix"), "C": st("C_dayshift") - st("B_all_anchors"),
           "C2": st("C2_rest_normal") - st("C_dayshift"), "D": st("D_stage3_bulk") - st("C2_rest_normal"),
           "E": st("E_perfect_tail") - st("D_stage3_bulk")}
    top_apt = d["oapt"].sort("delta__B_all_anchors").row(0, named=True)
    gi = d["gi"]
    gcs = gi.filter((pl.col("target") == "copy_SCHED") & ~pl.col("grouping").is_in(["month", "airport"]) & pl.col("auc_out_of_month").is_not_nan()).sort("auc_out_of_month", descending=True)
    gsum = {"best": gcs["auc_out_of_month"][0], "best_name": gcs["grouping"][0],
            "apt": gi.filter((pl.col("grouping") == "airport") & (pl.col("target") == "copy_SCHED"))["auc_out_of_month"][0],
            "aobt": gi.filter((pl.col("target") == "copy_AOBT3") & (pl.col("grouping") != "month"))["auc_out_of_month"].max(),
            "pres": gi.filter((pl.col("grouping") == "presence_pattern") & (pl.col("target") == "copy_any"))["auc_out_of_month"][0]}
    return f"""# V4 — Informe final: parada en la Etapa A

Generado por `src/v4_report.py`. Protocolo: `protocol.md` (guardado antes de auditar y de entrenar). Auditoría: `source_audit.md`.

## Veredicto

**La hipótesis de selección de fuentes no es viable.** BLOCK sí coincide con fuentes visibles en una fracción real de filas
(≈ 15–18 %; ≈ 5.8 % son copias no-SCHED sin doble conteo), pero esas filas son las que V3 ya predice bien. Un oráculo que
conoce exactamente qué filas copian una fuente no-SCHED mejora el RMSE ponderado jul/ene en **{a2t['delta_w_vs_v3']:+.1f} s**
(IC 95 % [{a2t['ci95_delta_w'][0]:+.1f}, {a2t['ci95_delta_w'][1]:+.1f}]), frente a los −15 s exigidos por el Gate A. Ni siquiera una compuerta perfecta
sobre todas las fuentes, SCHED incluida, baja del umbral de parada de 260 s ({f1(o['B_all_anchors']['rmse_w'])} s). Conforme al protocolo: no se
ejecutaron las Etapas B–E, no se entrenó ningún modelo V4, no existe `candidate_v4`, no se subió nada, no se hizo push y el
leaderboard no se usó para ninguna decisión.

## 1. Qué se ejecutó y qué no

| etapa | estado |
|---|---|
| Protocolo prerregistrado (`protocol.md`) | hecho, antes de cualquier cálculo |
| A · auditoría de fuentes (cobertura, atribución, confusión, aeropuerto, mes, operador, vuelo, callsign, stand, pista, ruta, presencia, resolución) | hecho → **Gate A no superado** |
| Oráculos suplementarios (no deciden el gate; etiquetados como cota optimista) | hecho, para cuantificar dónde está el margen |
| B · features V4 · C · C0–C5 · D · LOJO/LOJuO, 3 semillas · E · candidata | **no ejecutado** (parada prefijada) |

## 2. RMSE ponderado (julio 2025 y diciembre 2025 como proxy de enero; pesos del ranking 0.557 / 0.443)

C1–C5 no se ejecutaron. Como C0 se usa la predicción OOF ya existente de V3 (S4A_mix, folds de origen creciente), no un
reentrenamiento LOJO/LOJuO. Las filas siguientes son **oráculos**: conocen el target para decidir la compuerta; son cotas de lo
alcanzable con una compuerta perfecta, no resultados realizables. B→E son acumulativas.

{table(["configuración", "julio", "diciembre", "RMSE_w", "IC 95 % RMSE_w", "Δ vs C0 [IC 95 %]"], rows)}

Bootstrap: {orc['n_boot']} réplicas pareadas, bloques aeropuerto-día, estratificado por mes, recombinado con los pesos; semilla {orc['seed']}.
La razón de MSE de la Etapa 3 ({orc['stage3_bulk_mse_ratio']:.3f}) viene del diagnóstico V3 (`exp_capacity.json`).

## 3. Estabilidad por aeropuerto (RMSE ponderado jul/dic de cada aeropuerto; Δ de los oráculos frente a C0)

{table(["aeropuerto", "C0 (V3)", "Δ oráculo Gate A", "Δ todas las fuentes", "Δ hasta C2"], apt_rows)}

El oráculo del Gate A no llega a −15 s en {10 - orc['airports_A2_tau10_delta_le_-15s']} de 10 aeropuertos. El que más ganaría con una compuerta perfecta es
{top_apt['airport']} ({top_apt['delta__B_all_anchors']:+.1f} s con todas las fuentes; {top_apt['delta__C2_rest_normal']:+.1f} s hasta C2): es un problema de identificar la copia de
SCHED y la fecha desplazada, que V3 ya modela, no de añadir fuentes.

## 4. Columnas nuevas con señal real

Ninguna agrupación individual separa bien las fuentes. AUC fuera de mes (`source_audit.md` §8): la mejor llega a
{gsum['best']:.3f} para copias de SCHED ({gsum['best_name']}; {gsum['best'] - gsum['apt']:+.3f} sobre el aeropuerto solo), ≤ {gsum['aobt']:.3f} para copias de
AOBT_3; patrón de presencia ≈ {gsum['pres']:.2f}; la resolución de AOBT_3 no añade nada. **Son AUC univariados**: no acotan lo que
puede lograr un clasificador multivariable (el de V3 ya ronda 0.87 para la clase 1). Sin Etapas B/C no se midió la importancia
de las features V4 en un modelo; eso queda para V4b.

## 5. ARVT_1 / ARVT_3

No evaluadas: solo entraban como features de la variante withARVT en las Etapas B/C, que no se ejecutaron. No son fuentes de
BLOCK (son tiempos de llegada en destino, posteriores al despegue) y no intervienen en la auditoría ni en los oráculos.
No hay resultados "con y sin ARVT" que reportar.

## 6. Desviaciones y correcciones respecto al protocolo

- AUC de agrupaciones: el leave-one-out por fila estaba sesgado (§8 de la auditoría); se sustituyó por la tasa estimada fuera de
  mes antes de redactar. No afecta al gate.
- Oráculos suplementarios (umbrales ±30/60 s y escalera B→E) añadidos tras conocer el gate, solo para diagnóstico; el gate se
  decidió con la definición prerregistrada (±10 s), que no cambió.
- El oráculo del gate pasó a alinear filas con `align_by_id` (falla si algún MVT_ID falta o se duplica); el resultado es
  idéntico al de la primera ejecución.
- `models/experiments/v4/` quedó vacío: no se entrenó nada.

## 7. Qué implica (dónde está el margen, según los oráculos)

Con compuerta perfecta: saber qué filas copian una fuente (casi todo SCHED) vale {inc['B']:+.1f} s; la fecha desplazada,
{inc['C']:+.1f} s más; quitar la masa p1 que la mezcla reparte por prudencia en las filas normales, {inc['C2']:+.1f} s más; la
capacidad/escala de la Etapa 3, ≈ {inc['D']:+.1f} s; la cola > 1 h, {inc['E']:+.1f} s. Todo ese margen depende de **identificar**
qué filas son artefacto, no de qué fuente se copió. Las agrupaciones individuales identifican poco (AUC ≤ 0.72), pero eso no
acota un clasificador multivariable (V3 ≈ 0.87 para la clase 1); V4b lo prueba. Con compuerta perfecta sobre las fuentes el
techo es {f1(o['B_all_anchors']['rmse_w'])} s; una compuerta real queda por encima. Los pasos D y E de la escalera son oráculos/aproximaciones **no
alcanzables**; la banda 220–240 s solo aparece con identificación perfecta de todos los artefactos.

## 8. Integridad y cumplimiento

Verificación de solo lectura. Entorno: {ver['environment']}.

{table(["archivo / conjunto", "resultado"], ver_rows)}

Sin subidas al bucket, sin push, sin datos externos, sin uso del leaderboard. MVT_ID solo para alinear y FLIGHT_ID solo para
enlazar (no se usó). Los 14 Parquet, candidate_v1–v3 y sus JSON, los bundles finales y `_mac_tracked_md5.txt` no se tocaron.

## 9. Archivos

`src/v4_sources.py`, `src/v4_audit.py`, `src/v4_oracles.py`, `src/v4_report.py`, `tests/test_v4.py` y en esta carpeta:
`protocol.md`, `source_audit.md`, `report.md`, `decision.json`, `gate_a.json`, `oracles.json`, `oracle_by_airport.csv`,
`source_coverage.csv`, `source_attribution_by_airport.csv`, `source_attribution_by_month.csv`, `source_confusion.csv`,
`source_attribution_by_group.csv`, `source_group_informativeness.csv`, `timestamp_resolution.csv`, `aux_stand_anchors.csv`.
"""


def decision(d: dict, ver: dict) -> dict:
    g, o = d["gate"], d["orc"]["steps"]
    return {
        "version": "v4", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base_head": BASE_HEAD,
        "hypothesis": "BLOCK oculto = mezcla de expertos sobre fuentes visibles (AOBT_3, LOBT, EOBT_1, IOBT, SCHED, ±1 día, ninguna)",
        "gate_A": {"A1_copy_mass_non_sched_pct": g["A1_copy_mass_union_non_sched_pct"],
                   "A1_copy_mass_sum_non_sched_corrected_pct": g["A1_copy_mass_sum_non_sched_corrected_pct"], "A1_pass": g["A1_pass"],
                   "A2_rmse_w_v3": g["A2_oracle"]["weighted"]["rmse_s4a"], "A2_rmse_w_oracle": g["A2_oracle"]["weighted"]["rmse_oracle"],
                   "A2_delta_s": g["A2_oracle"]["weighted"]["delta"], "A2_delta_ci95": o["A2_nonSCHED_tau10"]["ci95_delta_w"],
                   "A2_threshold_s": g["A2_threshold_delta_s"], "A2_pass": g["A2_pass"], "pass": g["gate_A_pass"]},
        "decision": "STOP en la Etapa A: hipótesis no viable; no se ejecutan las Etapas B–E; no se genera candidate_v4"
                    if not g["gate_A_pass"] else "Gate A superado",
        "configs_run": [], "configs_not_run": ["C0", "C1", "C2", "C3", "C4", "C5"],
        "reference_c0_oof_origin_growing": {"rmse_w": o["V3_S4A_mix"]["rmse_w"], "rmse_jul": o["V3_S4A_mix"]["rmse_jul"],
                                            "rmse_dec_proxy_jan": o["V3_S4A_mix"]["rmse_dec"]},
        "oracle_ceilings_rmse_w": {k: {"rmse_w": v["rmse_w"], "delta_ci95": v["ci95_delta_w"]} for k, v in o.items() if k != "V3_S4A_mix"},
        "arvt": "no evaluado (Etapas B/C no ejecutadas); ARVT_1/3 no son fuentes de BLOCK",
        "candidate_v4": None, "uploaded_to_bucket": False, "pushed": False, "leaderboard_used_for_selection": False,
        "external_data_used": False, "protected_files_verification": ver,
    }


def main() -> None:
    d = load()
    ver = verify_protected()
    (OUT / "source_audit.md").write_text(source_audit_md(d), encoding="utf-8")
    (OUT / "report.md").write_text(report_md(d, ver), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision(d, ver), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"gate_A_pass": d["gate"]["gate_A_pass"], "verification": ver}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
