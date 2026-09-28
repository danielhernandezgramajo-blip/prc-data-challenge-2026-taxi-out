"""
FASE 1 — Validación de los datos del PRC Data Challenge 2026.

Comprobaciones:
 1. Existen exactamente 12 `training_2025-*.parquet`, `ranking.parquet` y `submitting.parquet`.
 2. Todos los Parquet abren (metadatos + lectura completa).
 3. Columnas y tipos idénticos entre los 13 archivos de 30 columnas; submitting = 2 columnas.
 4. Filas/columnas esperadas: training 4,167,797×30; ranking 689,534×30; submitting 344,841×2.
 5. `MVT_ID_mvt` único (dentro de training, dentro de ranking, y sin solape entre ambos).
 6. IDs y orden de `submitting.parquet` == filas DEP de `ranking.parquet`.
 7. Comprobaciones adicionales: target nulo en ranking-DEP, rangos temporales,
    identidad TAXITIME == MVT_TIME - BLOCK_TIME.

Uso:  python src/validate_data.py
Salida: outputs/reports/data_validation.md y outputs/tables/validation_*.csv
Los archivos originales solo se leen, nunca se escriben.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from config import (  # noqa: E402
    DATA_DIR, DEP, EXPECTED_N_COLS, EXPECTED_RANKING_ROWS, EXPECTED_SUBMITTING_COLS,
    EXPECTED_SUBMITTING_ROWS, EXPECTED_TRAINING_ROWS, ID_COL, PHASE_COL, RANKING_FILE,
    REPORTS_DIR, SUBMITTING_FILE, TABLES_DIR, TARGET, TRAINING_FILES, TRAINING_MONTHS,
)


class Report:
    """Acumula resultados de comprobaciones y los vuelca a Markdown."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def check(self, ok: bool, msg: str) -> bool:
        self.lines.append(f"- {'✅' if ok else '❌'} {msg}")
        if not ok:
            self.failures += 1
        print(("OK   " if ok else "FAIL "), msg)
        return ok

    def section(self, title: str) -> None:
        self.lines.append(f"\n## {title}\n")

    def note(self, text: str) -> None:
        self.lines.append(text)


def check_files(rep: Report) -> list[str]:
    rep.section("1. Inventario de archivos")
    present = sorted(p.name for p in DATA_DIR.glob("*.parquet"))
    train_present = [f for f in present if f.startswith("training_2025-")]
    rep.check(len(train_present) == 12, f"12 archivos training_2025-*.parquet (encontrados: {len(train_present)})")
    rep.check(set(train_present) == set(TRAINING_FILES), "Los 12 nombres de training coinciden con los meses esperados")
    rep.check(RANKING_FILE in present, "ranking.parquet presente")
    rep.check(SUBMITTING_FILE in present, "submitting.parquet presente")
    extra = sorted(set(present) - set(TRAINING_FILES) - {RANKING_FILE, SUBMITTING_FILE})
    rep.check(not extra, f"Sin Parquet inesperados en {DATA_DIR.name}/ (extra: {extra})")
    rep.note(f"\nCarpeta de datos: `{DATA_DIR}`\n")
    return present


def check_open_and_schema(rep: Report) -> dict[str, pl.DataFrame]:
    rep.section("2–4. Apertura, esquema y dimensiones")
    rows = []
    schemas: dict[str, list[tuple[str, str]]] = {}
    for f in [*TRAINING_FILES, RANKING_FILE, SUBMITTING_FILE]:
        path = DATA_DIR / f
        try:
            meta = pq.ParquetFile(path).metadata
            df = pl.read_parquet(path)  # lectura completa: garantiza que el archivo no está corrupto
            schemas[f] = [(c, str(t)) for c, t in df.schema.items()]
            rows.append({"file": f, "rows": df.height, "cols": df.width,
                         "row_groups": meta.num_row_groups, "size_mb": round(path.stat().st_size / 1e6, 1),
                         "opens": True})
            del df
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": f, "rows": None, "cols": None, "row_groups": None,
                         "size_mb": None, "opens": False})
            rep.check(False, f"{f} no abre: {exc}")
    inv = pl.DataFrame(rows)
    inv.write_csv(TABLES_DIR / "validation_file_inventory.csv")
    rep.check(inv["opens"].all(), "Todos los Parquet abren y se leen completos")

    # Esquemas
    ref = schemas[TRAINING_FILES[0]]
    same = all(schemas[f] == ref for f in [*TRAINING_FILES, RANKING_FILE])
    rep.check(same, "Columnas y tipos idénticos (nombre, orden y dtype) en los 12 training + ranking")
    sub_schema = schemas[SUBMITTING_FILE]
    rep.check(sub_schema == [(ID_COL, ref[0][1]), (TARGET, dict(ref)[TARGET])],
              f"submitting.parquet tiene exactamente [{ID_COL}, {TARGET}] con los mismos tipos que training")
    pl.DataFrame({"column": [c for c, _ in ref], "dtype": [t for _, t in ref]}).write_csv(TABLES_DIR / "validation_schema.csv")

    # Dimensiones
    train_rows = int(inv.filter(pl.col("file").str.starts_with("training_"))["rows"].sum())
    rank_rows = int(inv.filter(pl.col("file") == RANKING_FILE)["rows"][0])
    sub_rows = int(inv.filter(pl.col("file") == SUBMITTING_FILE)["rows"][0])
    rep.check(train_rows == EXPECTED_TRAINING_ROWS, f"Training: {train_rows:,} filas (esperado {EXPECTED_TRAINING_ROWS:,})")
    rep.check(all(inv.filter(pl.col("file") != SUBMITTING_FILE)["cols"] == EXPECTED_N_COLS), "Training y ranking: 30 columnas")
    rep.check(rank_rows == EXPECTED_RANKING_ROWS, f"Ranking: {rank_rows:,} filas (esperado {EXPECTED_RANKING_ROWS:,})")
    rep.check(sub_rows == EXPECTED_SUBMITTING_ROWS, f"Submitting: {sub_rows:,} filas (esperado {EXPECTED_SUBMITTING_ROWS:,})")
    rep.check(int(inv.filter(pl.col("file") == SUBMITTING_FILE)["cols"][0]) == EXPECTED_SUBMITTING_COLS, "Submitting: 2 columnas")

    rep.note("\n| archivo | filas | cols | MB |\n|---|---:|---:|---:|")
    for r in inv.iter_rows(named=True):
        rep.note(f"| {r['file']} | {r['rows']:,} | {r['cols']} | {r['size_mb']} |")
    return {}


def check_ids_and_alignment(rep: Report) -> None:
    rep.section("5–6. Unicidad de MVT_ID_mvt y alineación ranking ↔ submitting")
    # Solo columnas necesarias para no cargar 4 M filas completas en memoria
    tr = pl.concat([pl.scan_parquet(DATA_DIR / f).select(ID_COL, PHASE_COL) for f in TRAINING_FILES]).collect()
    rk = pl.read_parquet(DATA_DIR / RANKING_FILE, columns=[ID_COL, PHASE_COL, TARGET, "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
    sb = pl.read_parquet(DATA_DIR / SUBMITTING_FILE)

    rep.check(tr[ID_COL].null_count() == 0 and rk[ID_COL].null_count() == 0, "MVT_ID_mvt sin nulos")
    rep.check(tr[ID_COL].n_unique() == tr.height, f"MVT_ID_mvt único en training ({tr[ID_COL].n_unique():,} ids)")
    rep.check(rk[ID_COL].n_unique() == rk.height, f"MVT_ID_mvt único en ranking ({rk[ID_COL].n_unique():,} ids)")
    overlap = tr.join(rk.select(ID_COL), on=ID_COL, how="inner").height
    rep.check(overlap == 0, f"Sin solape de MVT_ID_mvt entre training y ranking (solape: {overlap})")
    # MVT_ID es float64 en los Parquet: comprobar que son enteros exactos
    is_int = bool((tr[ID_COL] == tr[ID_COL].round(0)).all() and (rk[ID_COL] == rk[ID_COL].round(0)).all())
    rep.check(is_int, "MVT_ID_mvt (float64) contiene únicamente valores enteros → puede convertirse a Int64 sin pérdida")

    rk_dep = rk.filter(pl.col(PHASE_COL) == DEP)
    rep.check(rk_dep.height == sb.height, f"Nº filas DEP en ranking ({rk_dep.height:,}) == filas de submitting ({sb.height:,})")
    same_order = rk_dep.height == sb.height and bool((rk_dep[ID_COL] == sb[ID_COL]).all())
    rep.check(same_order, "IDs y ORDEN de submitting coinciden exactamente con las filas DEP de ranking")
    same_set = set(rk_dep[ID_COL].to_list()) == set(sb[ID_COL].to_list())
    rep.check(same_set, "Conjunto de IDs de submitting == conjunto de IDs DEP de ranking")
    rep.check(sb[TARGET].null_count() == sb.height, "submitting.TAXITIME_SEC_mvt está completamente vacío (a rellenar)")
    rep.check(rk_dep[TARGET].null_count() == rk_dep.height, "ranking DEP: TAXITIME_SEC_mvt es nulo (target oculto)")
    rep.check(rk_dep["BLOCK_TIME_UTC_mvt"].null_count() == rk_dep.height,
              "ranking DEP: BLOCK_TIME_UTC_mvt es nulo (oculto: revelaría el target)")
    rk_arr = rk.filter(pl.col(PHASE_COL) != DEP)
    rep.check(rk_arr[TARGET].null_count() == 0, "ranking ARR: TAXITIME_SEC_mvt sí está informado (taxi-in disponible como contexto)")

    phases_tr = tr[PHASE_COL].value_counts().sort(PHASE_COL)
    phases_rk = rk[PHASE_COL].value_counts().sort(PHASE_COL)
    rep.note("\nDistribución de PHASE_mvt:\n")
    rep.note("| PHASE | training | ranking |\n|---|---:|---:|")
    for p in phases_tr[PHASE_COL].to_list():
        n_tr = phases_tr.filter(pl.col(PHASE_COL) == p)["count"][0]
        n_rk = phases_rk.filter(pl.col(PHASE_COL) == p)["count"][0]
        rep.note(f"| {p} | {n_tr:,} | {n_rk:,} |")


def check_temporal_and_target(rep: Report) -> None:
    rep.section("7. Comprobaciones adicionales (temporales y del target)")
    cols = [ID_COL, PHASE_COL, TARGET, "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    parts = []
    for f in TRAINING_FILES:
        d = pl.read_parquet(DATA_DIR / f, columns=cols)
        parts.append({
            "file": f,
            "rows": d.height,
            "rows_dep": d.filter(pl.col(PHASE_COL) == DEP).height,
            "mvt_min": d["MVT_TIME_UTC_mvt"].min(), "mvt_max": d["MVT_TIME_UTC_mvt"].max(),
            "target_null": d[TARGET].null_count(),
            "target_neg_dep": d.filter((pl.col(PHASE_COL) == DEP) & (pl.col(TARGET) < 0)).height,
            "target_zero_dep": d.filter((pl.col(PHASE_COL) == DEP) & (pl.col(TARGET) == 0)).height,
            "identity_holds_dep": bool(
                d.filter(pl.col(PHASE_COL) == DEP)
                 .select(((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds() == pl.col(TARGET)).all())
                 .item()),
        })
    months = pl.DataFrame(parts)
    months.write_csv(TABLES_DIR / "validation_months.csv")

    # Cada archivo cubre su mes. Se tolera hasta 60 s antes del inicio del mes: hay 1 fila
    # por archivo con MVT_TIME a las 23:59:5x del día anterior (la partición se hizo con
    # otra marca temporal o con redondeo). No es un error de datos, se documenta.
    from datetime import timedelta  # noqa: PLC0415
    tol = timedelta(seconds=60)
    boundary_rows = []
    for r, (a, b) in zip(months.iter_rows(named=True), TRAINING_MONTHS):
        start = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(b).replace(tzinfo=timezone.utc)
        ok = (r["mvt_min"] >= start - tol) and (r["mvt_max"] < end)
        boundary_rows.append(ok)
    rep.check(all(boundary_rows), "Cada archivo training_YYYY-MM cubre ese mes según MVT_TIME_UTC_mvt (tolerancia 60 s en el límite inferior)")
    n_before = sum(1 for r, (a, _) in zip(months.iter_rows(named=True), TRAINING_MONTHS)
                   if r["mvt_min"] < datetime.fromisoformat(a).replace(tzinfo=timezone.utc))
    rep.note(f"\n- ℹ️ {n_before} archivos contienen 1 fila con MVT_TIME_UTC_mvt unos segundos antes de la medianoche del mes "
             f"(p. ej. 2025-02-28 23:59:57 en el archivo de marzo). Sin impacto práctico; los cortes temporales del "
             f"backtesting se harán por MVT_TIME_UTC_mvt, no por archivo.")
    n_null = int(months["target_null"].sum())
    rep.note(f"- ℹ️ Training: {n_null} fila(s) con TAXITIME_SEC_mvt nulo en total (todas ARR; ver siguiente comprobación).")
    dep_null = int(sum(
        pl.read_parquet(DATA_DIR / f, columns=[PHASE_COL, TARGET]).filter((pl.col(PHASE_COL) == DEP) & pl.col(TARGET).is_null()).height
        for f in TRAINING_FILES))
    rep.check(dep_null == 0, f"Training DEP: TAXITIME_SEC_mvt sin nulos (nulos en DEP: {dep_null})")
    rep.check(bool(months["identity_holds_dep"].all()),
              "Training DEP: TAXITIME_SEC_mvt == (MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt) en el 100 % de filas "
              "→ el target es una función determinista de BLOCK_TIME (por eso está oculto en ranking)")
    neg = int(months["target_neg_dep"].sum()); zero = int(months["target_zero_dep"].sum())
    rep.note(f"\n- ℹ️ Training DEP: {neg:,} taxi-out negativos y {zero:,} iguales a 0 (se analizan en el EDA; no se eliminan aquí).")

    # Cadenas "NA" literales (artefacto de exportación desde R) que deberían ser nulos
    str_cols = [c for c, t in pl.read_parquet_schema(DATA_DIR / TRAINING_FILES[0]).items() if t == pl.String]
    na_tr = pl.concat([pl.scan_parquet(DATA_DIR / f).select(str_cols) for f in TRAINING_FILES]) \
              .select([(pl.col(c) == "NA").sum().alias(c) for c in str_cols]).collect()
    na_rk = pl.scan_parquet(DATA_DIR / RANKING_FILE).select(str_cols) \
              .select([(pl.col(c) == "NA").sum().alias(c) for c in str_cols]).collect()
    na_tbl = pl.DataFrame({"column": str_cols,
                           "na_literal_training": [int(na_tr[c][0]) for c in str_cols],
                           "na_literal_ranking": [int(na_rk[c][0]) for c in str_cols]})
    na_tbl.write_csv(TABLES_DIR / "validation_na_literals.csv")
    na_cols = na_tbl.filter((pl.col("na_literal_training") > 0) | (pl.col("na_literal_ranking") > 0))
    rep.note(f"- ⚠️ Cadenas literales \"NA\" (no nulos reales) en columnas de texto: "
             + (", ".join(f"{r['column']} ({r['na_literal_training']:,} train / {r['na_literal_ranking']:,} rank)"
                          for r in na_cols.iter_rows(named=True)) if na_cols.height else "ninguna")
             + ". Deben tratarse como ausentes en la preparación (`prepare_training.py`).")

    rk = pl.read_parquet(DATA_DIR / RANKING_FILE, columns=["MVT_TIME_UTC_mvt", PHASE_COL])
    rep.note(f"- ℹ️ Training cubre {months['mvt_min'].min()} → {months['mvt_max'].max()}; "
             f"ranking cubre {rk['MVT_TIME_UTC_mvt'].min()} → {rk['MVT_TIME_UTC_mvt'].max()} "
             f"(el ranking es posterior al training: la validación debe ser temporal).")
    rep.note("\n| mes | filas | filas DEP | negativos DEP | ceros DEP |\n|---|---:|---:|---:|---:|")
    for r in months.iter_rows(named=True):
        rep.note(f"| {r['file'][9:16]} | {r['rows']:,} | {r['rows_dep']:,} | {r['target_neg_dep']:,} | {r['target_zero_dep']:,} |")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.note("# Informe de validación de datos — PRC Data Challenge 2026\n")
    rep.note(f"Generado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · script: `src/validate_data.py`\n")
    check_files(rep)
    check_open_and_schema(rep)
    check_ids_and_alignment(rep)
    check_temporal_and_target(rep)
    rep.section("Resultado")
    rep.note("**TODAS LAS COMPROBACIONES PASAN.**" if rep.failures == 0 else f"**{rep.failures} comprobación(es) FALLAN.**")
    out = REPORTS_DIR / "data_validation.md"
    out.write_text("\n".join(rep.lines), encoding="utf-8")
    print(f"\nInforme escrito en {out.relative_to(out.parents[2])} — fallos: {rep.failures}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
