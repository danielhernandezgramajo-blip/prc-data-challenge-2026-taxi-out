"""
Utilidades de lectura (solo lectura) de los Parquet originales con Polars.

Notas:
* Los archivos fueron escritos desde R (`arrow.r.vctrs` en AIRCRAFT_OPERATOR_flt).
  Polars no conoce ese tipo de extensión; se carga como su tipo base (string).
* Nunca se escribe sobre los archivos originales.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

# Debe fijarse ANTES de importar polars
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl  # noqa: E402

from config import (  # noqa: E402
    DATA_DIR, DEP, PHASE_COL, RANKING_FILE, SUBMITTING_FILE, TRAINING_FILES,
)


def scan_training(columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """LazyFrame con los 12 meses de training concatenados (orden cronológico).

    Se añade la columna `source_file` para poder auditar de qué mes procede cada fila.
    """
    frames = []
    for f in TRAINING_FILES:
        lf = pl.scan_parquet(DATA_DIR / f)
        if columns is not None:
            lf = lf.select(list(columns))
        frames.append(lf.with_columns(pl.lit(f).alias("source_file")))
    return pl.concat(frames, how="vertical")


def scan_ranking(columns: Sequence[str] | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(DATA_DIR / RANKING_FILE)
    return lf.select(list(columns)) if columns is not None else lf


def read_submitting() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / SUBMITTING_FILE)


def only_dep(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filtra movimientos de salida (PHASE_mvt == 'DEP')."""
    return lf.filter(pl.col(PHASE_COL) == DEP)


def load_training_dep(columns: Iterable[str] | None = None) -> pl.DataFrame:
    """Training filtrado a DEP y materializado (≈2.1 M filas)."""
    cols = None if columns is None else list(dict.fromkeys([*columns, PHASE_COL]))
    return only_dep(scan_training(cols)).collect()


def load_ranking_dep(columns: Iterable[str] | None = None) -> pl.DataFrame:
    cols = None if columns is None else list(dict.fromkeys([*columns, PHASE_COL]))
    return only_dep(scan_ranking(cols)).collect()
