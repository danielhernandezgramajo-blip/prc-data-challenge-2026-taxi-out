import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def train_feats() -> pl.DataFrame:
    from prepare_training import load_features
    return load_features("train")


@pytest.fixture(scope="session")
def rank_feats() -> pl.DataFrame:
    from prepare_training import load_features
    return load_features("rank")


@pytest.fixture(scope="session")
def small_train(train_feats) -> pl.DataFrame:
    """Muestra pequeña y determinista (enero, 30 000 filas) para ajustes rápidos en los tests."""
    from config import TARGET
    jan = train_feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1).filter(pl.col(TARGET) > 0)
    return jan.head(30_000)
