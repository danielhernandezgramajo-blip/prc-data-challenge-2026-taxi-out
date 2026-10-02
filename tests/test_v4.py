"""
Pruebas V4 (Etapa A — auditoría de fuentes de BLOCK): atribución de fuentes, timestamps faltantes, desplazamiento ±1 día,
renormalización de probabilidades de la mezcla, ausencia de fuga entre folds mensuales (encoding fuera de mes), alineación
exacta de IDs, coherencia de las salidas y del Gate A, e integridad de los archivos protegidos (V1–V3, 14 Parquet, bundles).
La reproducibilidad de candidate_v4 no aplica: el Gate A falló y no se generó ninguna candidata (se comprueba que no existe).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs" / "experiments" / "v4"
T0 = datetime(2025, 7, 1, 10, 0, 0)
SRC_COLS = ["SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]


def _frame(rows: list[dict]) -> pl.DataFrame:
    """DataFrame sintético con las columnas visibles que usa anchor_frame (sin BLOCK ni TAXITIME)."""
    cols = {"MVT_ID_mvt": [r["id"] for r in rows], "MVT_TIME_UTC_mvt": [r["mvt"] for r in rows]}
    for c in SRC_COLS:
        cols[c] = [r.get(c) for r in rows]
    return pl.DataFrame(cols, schema={"MVT_ID_mvt": pl.Int64, "MVT_TIME_UTC_mvt": pl.Datetime("us"),
                                      **{c: pl.Datetime("us") for c in SRC_COLS}})


def _epoch(t: datetime) -> float:
    return pl.Series([t]).dt.epoch("s").cast(pl.Float64)[0]


# ---------------------------------------------------------------------------
# 1. Anclas: sin target, desplazamientos de ±1 día exactos, fuentes ausentes → NaN en las tres variantes
# ---------------------------------------------------------------------------
def test_anchor_frame_target_free_and_shifts():
    from v4_sources import ANCHORS, DAY, anchor_frame, anchor_matrix
    df = _frame([{"id": 1, "mvt": T0 + timedelta(minutes=20), "SCHED_TIME_UTC_mvt": T0, "IOBT_flt": T0, "LOBT_flt": T0,
                  "EOBT_1_flt": T0 + timedelta(minutes=2), "AOBT_3_flt": T0 + timedelta(minutes=5)},
                 {"id": 2, "mvt": T0 + timedelta(minutes=30), "SCHED_TIME_UTC_mvt": T0}])   # fuentes _flt ausentes
    af = anchor_frame(df)                                      # funciona sin BLOCK/TAXITIME: no puede leer el target
    assert not any(c in af.columns for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"))
    ts = anchor_matrix(af)
    assert ts.shape == (2, len(ANCHORS)) == (2, 15)
    for s in ("SCHED", "IOBT", "LOBT", "EOBT1", "AOBT3"):
        i, im, ip = ANCHORS.index(s), ANCHORS.index(s + "-1d"), ANCHORS.index(s + "+1d")
        assert ts[0, im] == ts[0, i] - DAY and ts[0, ip] == ts[0, i] + DAY
        if s != "SCHED":
            assert np.isnan(ts[1, [i, im, ip]]).all()
    assert ts[0, ANCHORS.index("AOBT3")] == _epoch(T0) + 300 and af["t_mvt"][1] == _epoch(T0) + 1800


# ---------------------------------------------------------------------------
# 2. Atribución: copia exacta, precedencia en empates, tolerancia, ambigüedad, desplazamiento de fecha
# ---------------------------------------------------------------------------
def test_attribution_rules():
    from v4_sources import ANCHORS, DAY, attribute
    K = len(ANCHORS); b = 1_750_000_000.0
    ts = np.full((6, K), np.nan)
    S, I, L, E, A = (ANCHORS.index(x) for x in ("SCHED", "IOBT", "LOBT", "EOBT1", "AOBT3"))
    ts[0, [S, I, L]] = b                        # empate exacto SCHED = IOBT = LOBT = BLOCK → gana SCHED (precedencia)
    ts[1, S] = b - 900; ts[1, A] = b - 5        # BLOCK = AOBT_3 + 5 s
    ts[2, S] = b - 900; ts[2, A] = b - 400      # nada cerca
    ts[3, S] = b - 20; ts[3, A] = b + 25        # dos anclas dentro de ±30 s que difieren 45 s → ambigua; gana la más cercana
    ts[4, S] = b + DAY; ts[4, A] = b + DAY - 300  # BLOCK = SCHED − 1 día exacto
    ts[4, ANCHORS.index("SCHED-1d")] = ts[4, S] - DAY; ts[4, ANCHORS.index("SCHED+1d")] = ts[4, S] + DAY
    ts[5, A] = b - DAY                          # BLOCK = AOBT_3 + 1 día
    ts[5, ANCHORS.index("AOBT3+1d")] = ts[5, A] + DAY; ts[5, ANCHORS.index("AOBT3-1d")] = ts[5, A] - DAY
    block = np.full(6, b)
    at = attribute(ts, block, 30.0)
    assert at["cls"][0] == S and at["n_within"][0] == 3 and not at["ambiguous"][0]   # anclas idénticas: no es ambigüedad
    assert at["cls"][1] == A and at["signed_d1"][1] == 5.0
    assert attribute(ts, block, 3.0)["cls"][1] == -1                                  # fuera de tolerancia
    assert at["cls"][2] == -1 and at["nearest"][2] == A and at["dist1"][2] == 400.0
    assert at["ambiguous"][3] and at["cls"][3] == S and at["second"][3] == A
    assert ANCHORS[at["cls"][4]] == "SCHED-1d" and at["dist1"][4] == 0.0
    assert ANCHORS[at["cls"][5]] == "AOBT3+1d"


def test_attribution_shares_deterministic_columns():
    from v4_audit import attribution_shares
    from v4_sources import ANCHORS
    cls = np.array([ANCHORS.index("AOBT3"), -1, ANCHORS.index("SCHED"), ANCHORS.index("EOBT1"), -1, ANCHORS.index("AOBT3")])
    keys = np.array(["B", "A", "B", "A", "B", "A"])
    out = attribution_shares(cls, keys, "apt")
    assert out.columns == ["apt", "n", "SCHED", "EOBT1", "AOBT3", "none"]          # orden canónico, no el del pivot
    assert out["apt"].to_list() == ["A", "B"] and out["n"].to_list() == [3, 3]
    assert np.allclose(out.select("SCHED", "EOBT1", "AOBT3", "none").sum_horizontal().to_numpy(), 100.0)
    rev = attribution_shares(cls[::-1], keys[::-1], "apt")
    assert rev.columns == out.columns and rev.equals(out)


# ---------------------------------------------------------------------------
# 3. Timestamps faltantes: se ignoran; fila sin ninguna fuente → sin atribución y sin errores
# ---------------------------------------------------------------------------
def test_attribution_missing_timestamps():
    from v4_sources import ANCHORS, attribute
    K = len(ANCHORS); b = 1_750_000_000.0
    ts = np.full((3, K), np.nan)
    ts[1, ANCHORS.index("EOBT1")] = b + 7
    ts[2, ANCHORS.index("SCHED")] = b - 1000; ts[2, ANCHORS.index("LOBT")] = b
    at = attribute(ts, np.full(3, b), 10.0)
    assert at["nearest"][0] == -1 and at["cls"][0] == -1 and np.isnan(at["dist1"][0]) and at["n_within"][0] == 0
    assert np.isnan(at["signed_d1"][0]) and at["second"][0] == -1 and np.isnan(at["dist2"][0])
    assert ANCHORS[at["cls"][1]] == "EOBT1" and at["second"][1] == -1               # una sola fuente disponible
    assert ANCHORS[at["cls"][2]] == "LOBT" and ANCHORS[at["second"][2]] == "SCHED"


# ---------------------------------------------------------------------------
# 4. Mezcla: E[BLOCK] con renormalización por fuentes ausentes, experto de fecha desplazada y corrección residual
# ---------------------------------------------------------------------------
def test_expected_block_renormalization():
    from v4_sources import DAY, expected_block
    ts = np.array([[1000.0, 2000.0], [1000.0, np.nan], [np.nan, np.nan], [500.0, 500.0]])
    none = np.array([3000.0, 3000.0, 3000.0, 500.0])
    p = np.array([[0.0, 1.0, 0.0], [0.5, 0.3, 0.2], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]])
    e = expected_block(p, ts, none)
    assert e[0] == 2000.0                                          # one-hot → ancla exacta
    assert np.isclose(e[1], (0.5 * 1000 + 0.2 * 3000) / 0.7)        # ancla ausente: p=0 y renormaliza sobre el resto
    assert e[2] == 3000.0                                          # toda la masa en anclas ausentes → 'none'
    assert np.isclose(e[3], 500.0)                                 # anclas y none iguales → invariante (pesos suman 1)
    pd_ = np.array([1.0, 0.0, 0.25, 0.0])
    e2 = expected_block(p, ts, none, p_ds=pd_)
    assert e2[0] == 3000.0 - DAY and e2[1] == e[1] and np.isclose(e2[2], 0.75 * 3000 + 0.25 * (3000 - DAY))
    r = np.array([[10.0, -20.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert expected_block(p, ts, none, resid=r)[0] == 1980.0        # corrección residual del experto
    with pytest.raises(ValueError):
        expected_block(-p, ts, none)
    with pytest.raises(ValueError):
        expected_block(p[:, :2], ts, none)


# ---------------------------------------------------------------------------
# 5. Sin fuga entre folds mensuales: el encoding de grupo de una fila no usa el target de su propio mes
# ---------------------------------------------------------------------------
def test_out_of_month_encoding_no_leakage():
    from v4_audit import out_of_month_rate
    rng = np.random.default_rng(0)
    n = 5000
    inv = rng.integers(0, 30, n); month = rng.integers(1, 13, n); y = (rng.random(n) < 0.1).astype(float)
    enc = out_of_month_rate(inv, month, y, prior=0.1)
    y2 = y.copy(); jul = month == 7; y2[jul] = 1.0 - y2[jul]           # se altera solo el target de julio
    enc2 = out_of_month_rate(inv, month, y2, prior=0.1)
    assert np.array_equal(enc[jul], enc2[jul])                         # julio no ve su propio target
    assert not np.array_equal(enc[~jul], enc2[~jul])                   # los demás meses sí ven julio (control)
    # agrupación = mes → solo el prior (no generaliza fuera de mes)
    assert np.allclose(out_of_month_rate(month - 1, month, y, prior=0.1), 0.1)
    # comprobación directa en una fila
    i = 0; m = (inv == inv[i]) & (month != month[i])
    assert np.isclose(enc[i], (y[m].sum() + 20 * 0.1) / (m.sum() + 20))


# ---------------------------------------------------------------------------
# 6. Alineación exacta de IDs (MVT_ID solo como clave)
# ---------------------------------------------------------------------------
def test_align_by_id_exact():
    from v4_sources import align_by_id
    af = pl.DataFrame({"MVT_ID_mvt": [30, 10, 20, 40], "v": [3.0, 1.0, 2.0, 4.0]})
    j = align_by_id(np.array([20.0, 40.0, 10.0]), af)                  # ids float enteros (como en meta__*.npz)
    assert j["MVT_ID_mvt"].to_list() == [20, 40, 10] and j["v"].to_list() == [2.0, 4.0, 1.0]
    for bad in (np.array([20, 99]), np.array([20, 20]), np.array([20.5])):
        with pytest.raises(ValueError):
            align_by_id(bad, af)
    with pytest.raises(ValueError):
        align_by_id(np.array([10]), pl.concat([af, af.head(1)]))


# ---------------------------------------------------------------------------
# 7. Masa de copia: recupera las copias inyectadas sobre un fondo uniforme
# ---------------------------------------------------------------------------
def test_copy_mass_recovers_injected_copies():
    from v4_sources import copy_mass
    rng = np.random.default_rng(1)
    bg = rng.integers(-300, 301, 200_000).astype(float); cp = np.zeros(4000)
    d = np.concatenate([bg, cp, np.full(500, np.nan)])
    assert abs(copy_mass(d, d.size) - 4000 / d.size) < 0.002


# ---------------------------------------------------------------------------
# 8. Salidas V4: Gate A coherente, oráculos coherentes con el gate, sin candidata ni modelos si el gate falla
# ---------------------------------------------------------------------------
def test_v4_outputs_consistent():
    if not (V4 / "gate_a.json").exists():
        pytest.skip("aún no se ha ejecutado v4_audit")
    g = json.loads((V4 / "gate_a.json").read_text())
    assert g["A2_pass"] == (g["A2_oracle"]["weighted"]["delta"] <= g["A2_threshold_delta_s"])
    assert g["A1_pass"] == (max(g["A1_copy_mass_union_non_sched_pct"], g["A1_copy_mass_sum_non_sched_corrected_pct"]) >= g["A1_threshold_pct"])
    assert g["gate_A_pass"] == (g["A1_pass"] and g["A2_pass"])
    for f in ("source_coverage.csv", "source_attribution_by_airport.csv", "source_attribution_by_month.csv", "source_confusion.csv"):
        assert (V4 / f).exists()
    if (V4 / "oracles.json").exists():
        o = json.loads((V4 / "oracles.json").read_text())["steps"]
        assert abs(o["A2_nonSCHED_tau10"]["rmse_w"] - g["A2_oracle"]["weighted"]["rmse_oracle"]) < 0.05
        assert abs(o["V3_S4A_mix"]["rmse_w"] - g["A2_oracle"]["weighted"]["rmse_s4a"]) < 0.05
        assert o["A2_nonSCHED_tau10"]["rmse_w"] >= o["A2_nonSCHED_tau30"]["rmse_w"] >= o["A2_nonSCHED_tau60"]["rmse_w"]
    if (V4 / "decision.json").exists():
        dec = json.loads((V4 / "decision.json").read_text())
        assert dec["gate_A"]["pass"] == g["gate_A_pass"]
        assert dec["uploaded_to_bucket"] is False and dec["pushed"] is False and dec["leaderboard_used_for_selection"] is False
        if not g["gate_A_pass"]:
            assert dec["candidate_v4"] is None and dec["configs_run"] == []
            assert not (ROOT / "submissions" / "candidate_v4.parquet").exists()
            assert not (ROOT / "submissions" / "candidate_v4.json").exists()
            mdir = ROOT / "models" / "experiments" / "v4"
            assert not mdir.exists() or not any(p.is_file() for p in mdir.rglob("*"))


# ---------------------------------------------------------------------------
# 9. Integridad de lo protegido (sustituye a la reproducibilidad de candidate_v4, que no existe)
# ---------------------------------------------------------------------------
# Hashes registrados ANTES de V4 (prefijos en protocol.md); una única fuente de verdad en src/v4_report.py
from v4_report import BUNDLE_DIRS, BUNDLES_JOINT_MD5, PARQUET_JOINT_SHA256, PROTECTED_SHA256  # noqa: E402


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_protected_files_unchanged():
    # Las predicciones (.parquet) se comprueban dondequiera que estén. Los metadatos (.json, _mac_tracked_md5.txt) solo en la
    # copia de trabajo git (la autoritativa): el espejo de cómputo en la nube no es un repo git y guarda borradores previos.
    authoritative = (ROOT / ".git").exists()
    present = {k: v for k, v in PROTECTED_SHA256.items() if (ROOT / k).exists() and (authoritative or k.endswith(".parquet"))}
    if not present:
        pytest.skip("archivos protegidos no presentes en este entorno")
    for rel, sha in present.items():
        assert _digest(ROOT / rel, "sha256") == sha, rel
    for v in (1, 2, 3):
        j = ROOT / "submissions" / f"candidate_v{v}.json"
        if j.exists() and authoritative:
            meta = json.loads(j.read_text())
            if "sha256" in meta:
                assert meta["sha256"] == PROTECTED_SHA256[f"submissions/candidate_v{v}.parquet"]


def test_challenge_parquets_and_bundles_unchanged():
    pq = sorted(p.name for p in ROOT.glob("*.parquet"))
    if len(pq) != 14:
        pytest.skip("los 14 Parquet no están en este entorno")
    listing = "".join(f"{_digest(ROOT / n, 'sha256')}  {n}\n" for n in pq)
    assert hashlib.sha256(listing.encode()).hexdigest() == PARQUET_JOINT_SHA256
    files = sorted(str(p.relative_to(ROOT)) for d in BUNDLE_DIRS for p in (ROOT / d).rglob("*") if p.is_file())
    if len(files) != 43:
        pytest.skip(f"bundles finales incompletos en este entorno ({len(files)} archivos)")
    listing = "".join(f"{_digest(ROOT / f, 'md5')}  {f}\n" for f in files)
    assert hashlib.md5(listing.encode()).hexdigest() == BUNDLES_JOINT_MD5
