"""
Utilidades compartidas de modelado (métricas, estrategias de outliers, codificación,
cronómetro). Sin lógica de modelos: eso vive en `train_baseline.py`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import polars as pl

from config import OUTLIER_HI_SEC, OUTLIER_STRATEGIES, TARGET

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def medae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.median(np.abs(y - p)))


def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    """Métricas para todas las filas, para el rango 'normal' (0 < y ≤ 3 h) y para los outliers."""
    normal = (y > 0) & (y <= OUTLIER_HI_SEC)
    out = {"n": int(y.size), "rmse": rmse(y, p), "mae": mae(y, p), "medae": medae(y, p),
           "bias": float(np.mean(p - y)), "n_pred_negative": int((p < 0).sum()),
           "n_normal": int(normal.sum()), "rmse_normal": rmse(y[normal], p[normal]), "mae_normal": mae(y[normal], p[normal]),
           "n_outlier": int((~normal).sum())}
    if (~normal).any():
        out["rmse_outlier"] = rmse(y[~normal], p[~normal]); out["mae_outlier"] = mae(y[~normal], p[~normal])
    else:
        out["rmse_outlier"] = np.nan; out["mae_outlier"] = np.nan
    return out


def metrics_by_airport(df: pl.DataFrame, pred: np.ndarray, key: str = "ADEP_mvt_key") -> pl.DataFrame:
    d = df.select(key, TARGET).with_columns(pl.Series("pred", pred))
    return (d.group_by(key).agg(pl.len().cast(pl.Int64).alias("n"),
                                ((pl.col(TARGET) - pl.col("pred")) ** 2).mean().sqrt().alias("rmse"),
                                (pl.col(TARGET) - pl.col("pred")).abs().mean().alias("mae"),
                                (pl.col(TARGET) - pl.col("pred")).abs().median().alias("medae"),
                                (pl.col("pred") - pl.col(TARGET)).mean().alias("bias"))
             .sort(key))


# ---------------------------------------------------------------------------
# Estrategias de outliers — SOLO sobre el bloque de entrenamiento
# ---------------------------------------------------------------------------
def apply_outlier_strategy(train: pl.DataFrame, strategy: str) -> tuple[pl.DataFrame, dict]:
    """Devuelve (train_modificado, info). El target de validación nunca se toca."""
    assert strategy in OUTLIER_STRATEGIES, strategy
    y = pl.col(TARGET)
    n0 = train.height
    if strategy == "original":
        out = train
    elif strategy == "drop_nonpositive":
        out = train.filter(y > 0)
    elif strategy == "drop_gt3h":
        out = train.filter(y <= OUTLIER_HI_SEC)
    elif strategy == "winsor_p01_p999":
        lo, hi = train[TARGET].quantile(0.001), train[TARGET].quantile(0.999)
        out = train.with_columns(y.clip(lo, hi).cast(pl.Int32))
        return out, {"strategy": strategy, "n_train_before": n0, "n_train_after": out.height, "winsor_lo": lo, "winsor_hi": hi}
    return out, {"strategy": strategy, "n_train_before": n0, "n_train_after": out.height, "winsor_lo": None, "winsor_hi": None}


# ---------------------------------------------------------------------------
# Codificación de categóricas para CatBoost (enteros estables; sin información del target)
# ---------------------------------------------------------------------------
class CategoryCodec:
    """Vocabulario string → int construido ÚNICAMENTE con el bloque de entrenamiento.

    No usa el target (no es target encoding). Las categorías que no estén en el vocabulario
    (stands/operadores nuevos en validación o en ranking) se codifican como -1, que CatBoost
    trata como una categoría más. El vocabulario se serializa completo en el bundle
    (`to_dict`/`from_dict`, `save`/`load`) para que la inferencia sea reproducible.
    """

    def __init__(self, cat_cols: list[str]) -> None:
        self.cat_cols = list(cat_cols)
        self.vocab: dict[str, dict[str, int]] = {}

    def fit(self, *frames: pl.DataFrame) -> "CategoryCodec":
        for c in self.cat_cols:
            vals = pl.concat([f.select(pl.col(c).cast(pl.String)) for f in frames])[c].unique().sort().to_list()
            self.vocab[c] = {v: i for i, v in enumerate(vals)}
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col(c).cast(pl.String).replace_strict(self.vocab[c], default=-1, return_dtype=pl.Int32).alias(c)
            for c in self.cat_cols
        ])

    # --- serialización ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"cat_cols": self.cat_cols, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "CategoryCodec":
        obj = cls(d["cat_cols"])
        obj.vocab = {c: {str(k): int(v) for k, v in m.items()} for c, m in d["vocab"].items()}
        return obj

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CategoryCodec":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __eq__(self, other) -> bool:
        return isinstance(other, CategoryCodec) and self.cat_cols == other.cat_cols and self.vocab == other.vocab


# ---------------------------------------------------------------------------
# Cronómetro
# ---------------------------------------------------------------------------
@contextmanager
def timer(store: dict, key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - t0, 2)
