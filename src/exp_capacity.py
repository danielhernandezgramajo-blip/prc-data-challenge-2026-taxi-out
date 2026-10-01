"""
Experimento diagnóstico (NO forma parte del pipeline de candidatas): ¿cuánto del error del grueso en julio 2025 es capacidad
del modelo o falta de features de escala, frente a ruido de etiqueta? Entrena con enero–junio (clase 0, y ≤ 3 h) y valida en
julio (clase 0). Tres configuraciones: A = la actual (1000 it, depth 6, lr 0.1); B = más capacidad (3000 it, depth 8, lr 0.05);
C = B + features de escala (llegada previa en el mismo stand: minutos desde su aterrizaje, su retraso de on-block, su taxi-in;
todas visibles en ranking) + FLIGHT_mvt como categórica. Salida: outputs/experiments/v3/exp_capacity.json
"""
from __future__ import annotations
import json, os, sys, time, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, polars as pl
from catboost import CatBoostRegressor, Pool
from config import FEATURE_CAT_COLS, FEATURE_NUM_COLS, ID_COL, TARGET, SEED
from prepare_training import load_features
from v2_protocol import regime_class_expr
from modeling_utils import CategoryCodec

OUT = "outputs/experiments/v3/exp_capacity.json"
t0 = time.time()
feats = load_features("train").with_columns(regime_class_expr(), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
# --- features de escala desde los movimientos crudos (DEP y ARR), solo columnas visibles en ranking
cols = [ID_COL, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "STAND_mvt", "FLIGHT_mvt"]
raw = pl.concat([pl.scan_parquet(f).select(cols) for f in sorted(glob.glob("training_2025-*.parquet"))]).collect()
raw = raw.with_columns(pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"),
                       pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t_mvt"), pl.col("STAND_mvt").cast(pl.String).fill_null("NA").alias("stand"))
arr = (raw.filter(pl.col("PHASE_mvt") == "ARR")
          .select("apt", "stand", pl.col("t_mvt").alias("t_arr"),
                  ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60).cast(pl.Float32).alias("prev_arr_delay_min"),
                  pl.col("TAXITIME_SEC_mvt").cast(pl.Float32).alias("prev_arr_taxi_in"))
          .sort("t_arr"))
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(ID_COL, "apt", "stand", "t_mvt", "FLIGHT_mvt").sort("t_mvt")
link = dep.join_asof(arr, left_on="t_mvt", right_on="t_arr", by=["apt", "stand"], strategy="backward", tolerance=24 * 3600)
link = link.with_columns(((pl.col("t_mvt") - pl.col("t_arr")) / 60).cast(pl.Float32).alias("min_since_prev_arr")).select(ID_COL, "min_since_prev_arr", "prev_arr_delay_min", "prev_arr_taxi_in", pl.col("FLIGHT_mvt").fill_null("UNK"))
feats = feats.join(link, on=ID_COL, how="left")
print(f"features listas {feats.shape} ({time.time()-t0:.0f}s)", flush=True)

tr = feats.filter((pl.col("mo") <= 6) & (pl.col("regime_class") == 0) & (pl.col(TARGET) > 0) & (pl.col(TARGET) <= 10800))
va = feats.filter((pl.col("mo") == 7) & (pl.col("regime_class") == 0) & (pl.col(TARGET) > 0))
print(f"train {tr.height:,} | val julio clase 0 {va.height:,}", flush=True)
EXTRA_NUM = ["min_since_prev_arr", "prev_arr_delay_min", "prev_arr_taxi_in"]
EXTRA_CAT = ["FLIGHT_mvt"]
configs = {
    "A_actual":      dict(iterations=1000, depth=6, learning_rate=0.1, num=FEATURE_NUM_COLS, cat=FEATURE_CAT_COLS),
    "B_capacidad":   dict(iterations=3000, depth=8, learning_rate=0.05, num=FEATURE_NUM_COLS, cat=FEATURE_CAT_COLS),
    "C_cap+escala":  dict(iterations=3000, depth=8, learning_rate=0.05, num=FEATURE_NUM_COLS + EXTRA_NUM, cat=FEATURE_CAT_COLS + EXTRA_CAT),
}
res = {}
yv = va[TARGET].to_numpy().astype(float); bulk = yv <= 3600
for name, cfg in configs.items():
    t1 = time.time()
    num, cat = cfg["num"], cfg["cat"]; fl = num + cat
    codec = CategoryCodec(cat).fit(tr)
    Xtr = codec.transform(tr.select(fl)).to_pandas(); Xva = codec.transform(va.select(fl)).to_pandas()
    ci = [fl.index(c) for c in cat]
    m = CatBoostRegressor(iterations=cfg["iterations"], depth=cfg["depth"], learning_rate=cfg["learning_rate"], loss_function="RMSE",
                          random_seed=SEED, thread_count=2, verbose=0, allow_writing_files=False, max_ctr_complexity=1)
    m.fit(Pool(Xtr, tr[TARGET].to_numpy().astype(np.float32), cat_features=ci))
    p = m.predict(Xva)
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    res[name] = {"rmse_clase0_julio": rm(p - yv), "rmse_grueso_le1h": rm((p - yv)[bulk]), "mae_grueso": float(np.mean(np.abs(p - yv)[bulk])),
                 "rmse_20_60min": rm((p - yv)[(yv > 1200) & (yv <= 3600)]), "fit_s": round(time.time() - t1), "n_feats": len(fl)}
    if name.startswith("C"):
        imp = dict(zip(fl, np.round(m.get_feature_importance(), 2).tolist())); res[name]["importance_extra"] = {k: imp[k] for k in EXTRA_NUM + EXTRA_CAT}
    print(name, res[name], flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
print("hecho", round((time.time() - t0) / 60, 1), "min")
