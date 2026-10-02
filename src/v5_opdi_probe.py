"""
V5 — Sondeo de viabilidad de OPDI (EUROCONTROL PRU + OpenSky Network, datos abiertos) como medida independiente del
off-block. Pregunta: ¿existe para nuestras salidas un evento ADS-B en tierra cuyo (MVT − evento) se parezca a TAXITIME mejor
que el AOBT_3 del NM? Si no, la ruta OPDI se descarta sin gastar más tiempo.

    .venv-mac/bin/python src/v5_opdi_probe.py                  # feb y ago 2025 (meses internos; ene/jul no se miran)
    .venv-mac/bin/python src/v5_opdi_probe.py --with-2026      # además, solo cobertura (sin target) en ene y jul 2026

No modifica los 14 Parquet originales ni sube nada. Descarga a data/external/opdi/raw/ (fuera de git), guarda el subconjunto
de nuestros 10 aeropuertos en data/external/opdi/ y el resumen en outputs/experiments/v5/opdi_probe.txt / .json.
Fuentes (patrones publicados en opdi.aero):
  flight_list   https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{YYYYMM}.parquet
  flight_events https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{d0}_{d0+10}.parquet
                (intervalos de 10 días anclados en 2022-01-01)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import warnings
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from config import DEP, OUTPUTS_DIR, PHASE_COL, PROJECT_ROOT  # noqa: E402

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
EXT = PROJECT_ROOT / "data" / "external" / "opdi"
RAW = EXT / "raw"
OUT = OUTPUTS_DIR / "experiments" / "v5"
ANCHOR = date(2022, 1, 1)
# un intervalo de 10 días completo dentro de cada mes (rejilla anclada en 2022-01-01)
PROBE_2025 = {"2025-02": date(2025, 2, 4), "2025-08": date(2025, 8, 3)}
PROBE_2026 = {"2026-01": date(2026, 1, 10), "2026-07": date(2026, 7, 9)}


def naive_utc(df: pl.DataFrame, col: str, alias: str | None = None) -> pl.DataFrame:
    """Columna de fecha-hora → Datetime('us') sin zona, en UTC (acepta naive, con zona o enteros epoch en s/ms/us)."""
    dt = df.schema[col]
    e = pl.col(col)
    if isinstance(dt, pl.Datetime):
        if dt.time_zone is not None:
            e = e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        e = e.cast(pl.Datetime("us"))
    elif dt in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64):
        mx = float(df[col].drop_nulls().max() or 0)
        unit = "s" if mx < 1e11 else ("ms" if mx < 1e14 else "us")
        e = pl.from_epoch(e.cast(pl.Int64), time_unit=unit).cast(pl.Datetime("us"))
    else:
        e = e.cast(pl.String).str.to_datetime(time_unit="us", time_zone=None, strict=False)
    return df.with_columns(e.alias(alias or col))


def norm_fl(fl: pl.DataFrame) -> pl.DataFrame:
    """Lista de vuelos OPDI → columnas id, FLT_ID, ADEP, ADES, first_seen (los ficheros reales vienen en minúscula)."""
    ren = {c: c.upper() for c in fl.columns if c.lower() in ("flt_id", "adep", "ades")}
    return fl.rename(ren)


def norm_ev(ev: pl.DataFrame) -> pl.DataFrame:
    """Eventos OPDI → `id` = identificador del VUELO (en v0.0.2 el vuelo es `flight_id` y `id` es el del evento)."""
    if "flight_id" in ev.columns:
        ev = ev.rename({"id": "event_id"}).rename({"flight_id": "id"})
    return ev


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url: str, dest: Path, tries: int = 3) -> Path | None:
    """Descarga con curl (certificados del sistema en macOS) y, si no hay curl, con urllib. None si el fichero no existe."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"ya descargado: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)"); return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log(f"descargando {url}")
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-fL", "--retry", str(tries), "--retry-delay", "5", "--progress-bar", "-o", str(tmp), url])
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.rename(dest); log(f"  {dest.stat().st_size / 1e6:.0f} MB"); return dest
        log(f"curl terminó con código {r.returncode} (22 = HTTP 4xx/5xx: el fichero no existe con ese nombre)")
        return None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prc-2026-optimistic-dragon/opdi-probe"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 22)
            tmp.rename(dest); log(f"  {dest.stat().st_size / 1e6:.0f} MB"); return dest
        except urllib.error.HTTPError as e:
            log(f"HTTP {e.code} en {url}")
            if e.code == 404:
                return None
        except Exception as e:  # red inestable: reintento
            log(f"error de descarga ({e}); reintento {k + 1}/{tries}"); time.sleep(5)
    return None


def grid_start(d: date) -> date:
    return ANCHOR + timedelta(days=((d - ANCHOR).days // 10) * 10)


def fetch_month(tag: str, d0: date) -> tuple[pl.DataFrame, pl.DataFrame, date, date] | None:
    """Subconjunto (lista de vuelos, eventos) de vuelos con ADEP o ADES en nuestros aeropuertos para [d0, d0 + 10).
    Si el nombre de la rejilla anclada en 2022-01-01 no existe, prueba intervalos alineados con el mes (1–11 y 11–21)."""
    ym = tag.replace("-", "")
    cached = sorted(EXT.glob(f"flight_events_{ym}*_10apt.parquet"))
    fl_out = EXT / f"flight_list_{ym}_10apt.parquet"
    if fl_out.exists() and cached:
        d0c = datetime.strptime(cached[0].name.split("_")[2], "%Y%m%d").date()
        return norm_fl(pl.read_parquet(fl_out)), norm_ev(pl.read_parquet(cached[0])), d0c, d0c + timedelta(days=10)
    fl_raw = download(f"{BASE}/flight_list/flight_list_{ym}.parquet", RAW / f"flight_list_{ym}.parquet")
    if fl_raw is None:
        log(f"{tag}: no se pudo descargar la lista de vuelos"); return None
    first = date(int(ym[:4]), int(ym[4:]), 1)
    ev_raw = None
    for a in (d0, first, first + timedelta(days=10)):
        b = a + timedelta(days=10)
        ev_raw = download(f"{BASE}/flight_events/flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet", RAW / f"flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet")
        if ev_raw is not None:
            d0 = a; break
    if ev_raw is None:
        log(f"{tag}: no se encontró el fichero de eventos (¿cambió el patrón de nombres en opdi.aero?)"); return None
    d1 = d0 + timedelta(days=10)
    fl = norm_fl(pl.read_parquet(fl_raw))
    log(f"{tag}: flight_list {fl.shape} columnas {fl.columns}")
    fl = fl.filter(pl.col("ADEP").is_in(APTS) | pl.col("ADES").is_in(APTS))
    key = "flight_id" if "flight_id" in pl.scan_parquet(ev_raw).collect_schema().names() else "id"
    q = pl.scan_parquet(ev_raw).join(fl.lazy().select(pl.col("id").alias(key)).unique(), on=key, how="semi")
    try:
        ev = q.collect(engine="streaming")      # poca memoria con ficheros grandes
    except Exception:
        ev = q.collect()
    log(f"{tag}: eventos de nuestros vuelos {ev.shape} columnas {ev.columns}")
    EXT.mkdir(parents=True, exist_ok=True)
    fl.write_parquet(fl_out); ev.write_parquet(EXT / f"flight_events_{d0:%Y%m%d}_10apt.parquet")
    return fl, norm_ev(ev), d0, d1


def our_dep(d0: date, d1: date, split: str) -> pl.DataFrame:
    from io_utils import scan_ranking, scan_training
    cols = [PHASE_COL, "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "AOBT_3_flt"]
    if split == "train":
        cols.append("TAXITIME_SEC_mvt")
    lf = (scan_training if split == "train" else scan_ranking)(cols)
    d = lf.filter(pl.col(PHASE_COL) == DEP).collect()
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"):
        d = naive_utc(d, c)
    t0 = datetime(d0.year, d0.month, d0.day); t1 = datetime(d1.year, d1.month, d1.day)
    d = d.filter((pl.col("MVT_TIME_UTC_mvt") >= t0) & (pl.col("MVT_TIME_UTC_mvt") < t1))
    return d.with_columns(pl.col("CALLSIGN_flt").cast(pl.String).str.strip_chars().str.to_uppercase().alias("cs"),
                          pl.col("MVT_TIME_UTC_mvt").alias("mvt"))


def match(dep: pl.DataFrame, fl: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """Empareja cada salida con un vuelo OPDI del mismo ADEP y callsign: con despegue OPDI a ≤ 15 min de MVT o, si el vuelo
    OPDI no tiene evento de despegue, con first_seen en [MVT − 4 h, MVT + 10 min]. Se prefiere el emparejamiento por despegue."""
    fl = naive_utc(fl.with_columns(pl.col("FLT_ID").cast(pl.String).str.strip_chars().str.to_uppercase().alias("cs")), "first_seen")
    ev = naive_utc(ev, "event_time")
    types = [t for t in ev["type"].cast(pl.String).unique().to_list() if t]
    to_types = [t for t in types if "take" in t.lower() and "off" in t.lower()]
    log(f"tipos de evento ({len(types)}): {sorted(types)[:40]} | despegue: {to_types}")
    to = (ev.filter(pl.col("type").cast(pl.String).is_in(to_types)).group_by("id")
            .agg(pl.col("event_time").min().alias("t_takeoff_opdi")))
    cand = (dep.filter(pl.col("cs").is_not_null()).join(fl.select("id", "cs", "ADEP", "first_seen"), left_on=["ADEP_mvt", "cs"],
                                                        right_on=["ADEP", "cs"], how="inner")
               .join(to, on="id", how="left")
               .with_columns((pl.col("t_takeoff_opdi") - pl.col("mvt")).dt.total_seconds().abs().alias("gap_to"),
                             (pl.col("mvt") - pl.col("first_seen")).dt.total_seconds().alias("fs_before")))
    ok_to = pl.col("gap_to").is_not_null() & (pl.col("gap_to") <= 900)
    ok_fs = pl.col("gap_to").is_null() & (pl.col("fs_before") >= -600) & (pl.col("fs_before") <= 4 * 3600)
    cand = cand.filter(ok_to | ok_fs).with_columns(pl.when(ok_to).then(pl.col("gap_to")).otherwise(100_000 + pl.col("fs_before").abs()).alias("rank"))
    return cand.sort("rank").unique(subset=["MVT_ID_mvt"], keep="first").unique(subset=["id"], keep="first")


def evaluate(tag: str, dep: pl.DataFrame, best: pl.DataFrame, ev: pl.DataFrame, with_target: bool) -> dict:
    res: dict = {"tag": tag, "dep_rows": dep.height, "matched": best.height,
                 "matched_by_takeoff_event": int(best["t_takeoff_opdi"].is_not_null().sum())}
    res["match_rate_by_airport"] = {r["ADEP_mvt"]: round(r["rate"], 3) for r in
                                    dep.join(best.select("MVT_ID_mvt").with_columns(pl.lit(1).alias("m")), on="MVT_ID_mvt", how="left")
                                       .group_by("ADEP_mvt").agg(pl.col("m").fill_null(0).mean().alias("rate")).sort("ADEP_mvt")
                                       .iter_rows(named=True)}
    ev = naive_utc(ev, "event_time")
    # first_seen de la lista de vuelos como evento adicional (opdi.aero lo describe como primera detección del vuelo)
    fs = best.select("id", pl.lit("first_seen(flight_list)").alias("type"), pl.col("first_seen").alias("event_time"))
    e = (pl.concat([ev.select("id", pl.col("type").cast(pl.String), pl.col("event_time")), fs], how="vertical_relaxed")
           .join(best.select("id", "ADEP_mvt", "mvt", *(["TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"] if with_target else [])), on="id")
           .with_columns(((pl.col("mvt") - pl.col("event_time")).dt.total_seconds()).alias("d")))
    e = e.filter((pl.col("d") >= -600) & (pl.col("d") <= 4 * 3600))      # eventos en el aeropuerto de salida, antes del despegue
    # un evento por (vuelo, tipo): el más temprano dentro de la ventana (off-block / inicio de rodaje)
    e = e.sort("event_time").unique(subset=["id", "type"], keep="first")
    res["event_types_near_departure"] = {r["type"]: int(r["len"]) for r in e.group_by("type").len().sort("len", descending=True).iter_rows(named=True)}
    if not with_target:
        return res
    y = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
    s = (pl.col("mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
    e = e.filter(((y - s).abs() > 10) & (y > 0) & (y <= 3600))           # clase normal (BLOCK ≠ SCHED), y ≤ 1 h
    e = e.with_columns((pl.col("mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("d_nm"))
    n0 = dep.filter((pl.col("TAXITIME_SEC_mvt") > 0) & (pl.col("TAXITIME_SEC_mvt") <= 3600)).height
    stats = {}
    for t, g in e.group_by("type"):
        t = t[0]
        if g.height < 200:
            continue
        g = g.filter(pl.col("d").is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null())
        yy = g["TAXITIME_SEC_mvt"].to_numpy().astype(float); dd = g["d"].to_numpy().astype(float)
        if yy.size < 200:
            continue
        r = yy - dd; off = float(np.median(r)); a = np.abs(r - off)
        nm = g["d_nm"].to_numpy().astype(float); ok = np.isfinite(nm); rn = yy[ok] - nm[ok]; an = np.abs(rn - np.median(rn)) if ok.any() else rn
        try:
            b = np.polyfit(dd, yy, 1) if np.std(dd) > 0 else np.array([0.0, float(np.mean(yy))])
        except Exception:
            b = np.array([0.0, float(np.mean(yy))])
        stats[t] = {"n": g.height, "coverage_of_dep": round(g.height / max(n0, 1), 3), "median_y_minus_d": round(off, 1),
                    "share_within_30s": round(float((a <= 30).mean()), 3), "share_within_60s": round(float((a <= 60).mean()), 3),
                    "share_within_120s": round(float((a <= 120).mean()), 3), "corr_y_d": round(float(np.corrcoef(yy, dd)[0, 1]), 3) if np.std(dd) > 0 else None,
                    "rmse_linear_fit": round(float(np.sqrt(np.mean((yy - np.polyval(b, dd)) ** 2))), 1),
                    "rmse_airport_mean": round(float(np.sqrt(np.mean((yy - g.with_columns(pl.col("TAXITIME_SEC_mvt").mean().over("ADEP_mvt").alias("m"))["m"].to_numpy()) ** 2))), 1),
                    "NM_AOBT3_same_rows_within_60s": round(float((an <= 60).mean()), 3) if ok.any() else None,
                    "by_airport_within_60s": {k: round(float(v), 3) for k, v in
                                              g.with_columns(((y - pl.col("d")) - off).abs().le(60).alias("w")).group_by("ADEP_mvt")
                                               .agg(pl.col("w").mean()).sort("ADEP_mvt").iter_rows()}}
    res["vs_target_class0_le1h"] = dict(sorted(stats.items(), key=lambda kv: -kv[1]["share_within_60s"]))
    return res


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-2026", action="store_true", help="también cobertura en ene/jul 2026 (sin target)")
    ap.add_argument("--keep-raw", action="store_true", help="no borrar los ficheros completos descargados")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    plan = [(k, v, "train") for k, v in PROBE_2025.items()] + ([(k, v, "rank") for k, v in PROBE_2026.items()] if args.with_2026 else [])
    for tag, d0, split in plan:
        got = fetch_month(tag, d0)
        if got is None:
            report.append({"tag": tag, "error": "descarga fallida"}); continue
        fl, ev, d0, d1 = got
        dep = our_dep(d0, d1, split)
        best = match(dep, fl, ev)
        r = evaluate(tag, dep, best, ev, with_target=(split == "train"))
        r["period"] = f"{d0} .. {d1 - timedelta(days=1)}"
        report.append(r)
        log(json.dumps({k: v for k, v in r.items() if k != "vs_target_class0_le1h"}, ensure_ascii=False)[:1500])
        for t, st in list(r.get("vs_target_class0_le1h", {}).items())[:6]:
            log(f"  {t}: {json.dumps(st, ensure_ascii=False)}")
    (OUT / "opdi_probe.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    with open(OUT / "opdi_probe.txt", "w", encoding="utf-8") as fh:
        for r in report:
            fh.write(json.dumps(r, indent=1, ensure_ascii=False, default=str) + "\n")
    if not args.keep_raw and RAW.exists():
        for f in RAW.glob("*.parquet"):
            f.unlink()
        log("ficheros completos borrados (se conserva el subconjunto de nuestros 10 aeropuertos en data/external/opdi/)")
    log(f"Resumen en {OUT / 'opdi_probe.txt'} — no se ha subido nada.")


if __name__ == "__main__":
    main()
