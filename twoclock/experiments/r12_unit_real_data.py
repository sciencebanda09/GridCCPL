"""R12 -- Unit-Level Real Market Data Discovery for Stage 3 (NYISO 2024 + EPA CEMS).

Tests Stage 3 inverse wear-price optimization at the *individual generating unit*
level using 100% official public U.S. datasets (zero API keys):
  1. EPA CEMS 2024 hourly unit-level gross load, heat input, and CO2 emissions
     (Catalyst Cooperative PUDL public AWS S3 Open Data).
  2. EIA-860 plant/generator metadata, EPA-EIA crosswalk, and EIA-923 monthly
     delivered fuel prices (Catalyst Cooperative PUDL public AWS S3 Open Data).
  3. NYISO 2024 hourly Day-Ahead generator-nodal (P-2B) and zonal (P-2A) LBMPs.

Protocol (strictly outcome-independent):
  * Cohort defined purely on physical / EIA / metering criteria BEFORE inverse
    estimation (28 non-cogen dispatchable NYISO thermal units across 14 plants,
    152,592 unit-hours; +6 EIA-860 cogen-tagged merchant CCGTs in sensitivity).
  * Exact single-unit price-taking LP solved via persistent warm-started HiGHS
    over all non-overlapping 168-h weekly windows (>=80% online, n=778), all
    72-h windows (parallel to R5), and full-year online series.
  * Mandatory R5-grade statistical power, placebo & confounder suite:
      - Positive controls w_true in {15, 30, 60, 120} $/MWh with 24-h
        block-bootstrapped empirical residuals (directly tests unit-level power).
      - Null placebos (w_true = 0) with 24-h block-bootstrapped, i.i.d.
        permuted, and Gaussian residuals.
      - Within-window LBMP-shuffle placebo (paired SSE & identification check).
      - Equal-DOF (1 extra param) 4-model horse race: Blind LP vs Wear-Kink LP
        vs Smooth Quadratic Cost QP (HiGHS QP) vs Constant Operational Ceiling LP
        (evaluated on BOTH real data AND planted w_true = 30 positive control).
      - Non-parametric margin-response & bunching diagnostics, ambient summer
        capacity derating (seasonal vs annual P99.5), winter cold-snap fuel/oil
        switching (via CEMS CO2/heat), duct-firing incremental heat-rate kink,
        nodal vs zonal LBMP, sigma in {0.80, 0.85, 0.90}, cost shift +/-$10/MWh.

Run (from the folder containing twoclock/):
    python -m twoclock.experiments.r12_unit_real_data
    python -m twoclock.experiments.r12_unit_real_data --plot
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

# Prevent OpenBLAS / OMP thread explosion inside Windows spawned worker pools
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.optimize._highspy._core as _h
from scipy.sparse import csc_matrix
from scipy import stats

# Package / path setup (works as module or standalone script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PKG_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(PKG_DIR, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "r12_unit_real_data.json")
OUT_FIG_PDF = os.path.join(RESULTS_DIR, "fig_r12_unit_real_data.pdf")
OUT_FIG_PNG = os.path.join(RESULTS_DIR, "fig_r12_unit_real_data.png")

RGGI_USD_PER_SHORT_TON = 20.0
W_MAX = 300.0
W_GRID = np.concatenate([
    np.arange(0, 101, 1),
    np.arange(102, 201, 2),
    np.arange(205, 301, 5),
]).astype(float)
G_GRID = np.concatenate([[0.0], np.geomspace(1.0, 300.0, 25)])
TH_GRID = np.linspace(0.65, 1.00, 36)
TOL = 1.05
POS_W_LIST = (15.0, 30.0, 60.0, 120.0)

# Verified 1-to-1 engineering crosswalk: (plant_id_eia, unit_id) -> (NYISO gen node, NYISO zone, tech)
# Constructed strictly from EIA-860/EPA-CAMD/NYISO PTID names before any inverse estimation.
NYISO_NODE_MAP = {
    (2490, "20"):     ("ARTHUR_KILL_2", "N.Y.C.", "ST"),
    (2490, "30"):     ("ARTHUR_KILL_3", "N.Y.C.", "ST"),
    (2493, "1"):      ("EAST_RIVER___1", "N.Y.C.", "GT"),
    (2493, "2"):      ("EAST_RIVER___2", "N.Y.C.", "GT"),
    (2500, "UCC001"): ("RAVENSWOOD___4", "N.Y.C.", "CCGT"),
    (2511, "10"):     ("BARRETT___1", "LONGIL", "ST"),
    (2511, "20"):     ("BARRETT___2", "LONGIL", "ST"),
    (2516, "1"):      ("NORTHPORT___1", "LONGIL", "ST"),
    (2516, "2"):      ("NORTHPORT___2", "LONGIL", "ST"),
    (2516, "3"):      ("NORTHPORT___3", "LONGIL", "ST"),
    (2516, "4"):      ("NORTHPORT___4", "LONGIL", "ST"),
    (2517, "4"):      ("PORT_JEFF_4", "LONGIL", "ST"),
    (2539, "10001"):  ("BETHLEHEM___GS1", "CAPITL", "CCGT"),
    (2539, "10002"):  ("BETHLEHEM___GS2", "CAPITL", "CCGT"),
    (2539, "10003"):  ("BETHLEHEM___GS3", "CAPITL", "CCGT"),
    (8906, "AST3"):   ("ASTORIA___3", "N.Y.C.", "ST"),
    (8906, "AST5"):   ("ASTORIA___5", "N.Y.C.", "ST"),
    (54547, "1"):     ("SITHE_IND_GS1", "CENTRL", "CCGT"),
    (54547, "2"):     ("SITHE_IND_GS2", "CENTRL", "CCGT"),
    (54547, "3"):     ("SITHE_IND_GS3", "CENTRL", "CCGT"),
    (54547, "4"):     ("SITHE_IND_GS4", "CENTRL", "CCGT"),
    (54914, "1"):     ("BROOKLYN_NAVY_YARD", "N.Y.C.", "CCGT"),
    (54914, "2"):     ("BROOKLYN_NAVY_YARD", "N.Y.C.", "CCGT"),
    (55375, "CT1"):   ("ASTORIA_EAST_ENERGY_CC1", "N.Y.C.", "CCGT"),
    (55375, "CT2"):   ("ASTORIA_EAST_ENERGY_CC2", "N.Y.C.", "CCGT"),
    (55405, "1"):     ("ATHENS_STG_1", "HUD VL", "CCGT"),
    (55405, "2"):     ("ATHENS_STG_2", "HUD VL", "CCGT"),
    (55405, "3"):     ("ATHENS_STG_3", "HUD VL", "CCGT"),
    (56196, "CTG7A"): ("NYPA___ASTORIA_CC1", "N.Y.C.", "CCGT"),
    (56196, "CTG7B"): ("NYPA___ASTORIA_CC2", "N.Y.C.", "CCGT"),
    (56234, "1"):     ("CAITHNESS_CC_1", "LONGIL", "CCGT"),
    (56259, "CT-1"):  ("EMPIRE_CC_1", "CAPITL", "CCGT"),
    (56259, "CT-2"):  ("EMPIRE_CC_2", "CAPITL", "CCGT"),
    (56940, "1"):     ("CPV_VALLEY___CC1", "HUD VL", "CCGT"),
    (56940, "2"):     ("CPV_VALLEY___CC2", "HUD VL", "CCGT"),
    (57185, "U001"):  ("CRICKET___VALLEY_CC1", "HUD VL", "CCGT"),
    (57185, "U002"):  ("CRICKET___VALLEY_CC2", "HUD VL", "CCGT"),
    (57185, "U003"):  ("CRICKET___VALLEY_CC3", "HUD VL", "CCGT"),
    (57664, "CT3"):   ("AST_ENERGY_2_CC3", "N.Y.C.", "CCGT"),
    (57664, "CT4"):   ("AST_ENERGY_2_CC4", "N.Y.C.", "CCGT"),
}

TWIN_BOILER_MAP = {
    (8906, "31RH"): "AST3",
    (8906, "32SH"): "AST3",
    (8906, "51RH"): "AST5",
    (8906, "52SH"): "AST5",
}


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# Step 1: Data acquisition & local caching (PUDL S3 + NYISO MIS)
# ---------------------------------------------------------------------------

def ensure_data_cached() -> None:
    """Download and cache NY 2024 EPA CEMS, EIA-860/923, and NYISO DAM LBMPs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cems_p = os.path.join(DATA_DIR, "cems_ny_2024.parquet")
    ent_p = os.path.join(DATA_DIR, "eia_entity_plants_ny.parquet")
    scd_p = os.path.join(DATA_DIR, "eia860_scd_plants_ny.parquet")
    gen_p = os.path.join(DATA_DIR, "eia860_scd_generators_ny.parquet")
    assn_p = os.path.join(DATA_DIR, "epa_assn_ny.parquet")
    fuel_p = os.path.join(DATA_DIR, "eia923_fuel_ny.parquet")
    zone_p = os.path.join(DATA_DIR, "nyiso_damlbmp_zone_2024.parquet")
    ngen_p = os.path.join(DATA_DIR, "nyiso_damlbmp_gen_2024.parquet")

    pudl_needed = [cems_p, ent_p, scd_p, gen_p, assn_p, fuel_p]
    if any(not os.path.exists(p) for p in pudl_needed):
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        base = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable"
        if not os.path.exists(cems_p):
            log("Downloading NY 2024 EPA CEMS from Catalyst PUDL S3...")
            con.execute(f"""
                COPY (
                    SELECT state, year, plant_id_eia, plant_id_epa, emissions_unit_id_epa,
                           operating_datetime_utc, operating_time_hours, gross_load_mw,
                           steam_load_lbs, heat_content_mmbtu, co2_mass_tons, so2_mass_lbs, nox_mass_lbs
                    FROM read_parquet('{base}/core_epacems__hourly_emissions.parquet')
                    WHERE state = 'NY' AND year = 2024
                ) TO '{cems_p}' (FORMAT PARQUET)
            """)
        if not os.path.exists(ent_p):
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{base}/core_eia__entity_plants.parquet')
                      WHERE state = 'NY') TO '{ent_p}' (FORMAT PARQUET)
            """)
        if not os.path.exists(scd_p):
            con.execute(f"""
                COPY (SELECT s.* FROM read_parquet('{base}/core_eia860__scd_plants.parquet') s
                      SEMI JOIN read_parquet('{ent_p}') e ON s.plant_id_eia = e.plant_id_eia
                      WHERE s.report_date >= DATE '2022-01-01') TO '{scd_p}' (FORMAT PARQUET)
            """)
        if not os.path.exists(gen_p):
            con.execute(f"""
                COPY (SELECT g.* FROM read_parquet('{base}/core_eia860__scd_generators.parquet') g
                      SEMI JOIN read_parquet('{ent_p}') e ON g.plant_id_eia = e.plant_id_eia
                      WHERE g.report_date >= DATE '2022-01-01') TO '{gen_p}' (FORMAT PARQUET)
            """)
        if not os.path.exists(assn_p):
            con.execute(f"""
                COPY (SELECT a.* FROM read_parquet('{base}/core_epa__assn_eia_epacamd.parquet') a
                      SEMI JOIN read_parquet('{ent_p}') e ON a.plant_id_eia = e.plant_id_eia)
                TO '{assn_p}' (FORMAT PARQUET)
            """)
        if not os.path.exists(fuel_p):
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{base}/out_eia923__monthly_fuel_receipts_costs.parquet')
                      WHERE state = 'NY' AND report_date >= DATE '2023-01-01')
                TO '{fuel_p}' (FORMAT PARQUET)
            """)

    for kind, out_pq in (("zone", zone_p), ("gen", ngen_p)):
        if os.path.exists(out_pq):
            continue
        log(f"Downloading NYISO 2024 DAM LBMP ({kind})...")
        dfs = []
        for mm in range(1, 13):
            url = f"https://mis.nyiso.com/public/csv/damlbmp/2024{mm:02d}01damlbmp_{kind}_csv.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            zf = zipfile.ZipFile(io.BytesIO(raw))
            for name in sorted(zf.namelist()):
                if name.endswith(".csv"):
                    with zf.open(name) as fh:
                        dfs.append(pd.read_csv(fh))
        pd.concat(dfs, ignore_index=True).to_parquet(out_pq, index=False)


def parse_nyiso_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NYISO America/New_York wall-clock timestamps (DST-aware) to naive UTC."""
    out = []
    for name, g in df.groupby("Name", sort=False):
        g = g.sort_index().copy()
        dt_naive = pd.to_datetime(g["Time Stamp"], format="%m/%d/%Y %H:%M")
        # On fall-back DST day (Nov 3, 2024), first 01:00 is EDT (DST=True), second is EST (DST=False).
        is_dup = dt_naive.duplicated(keep="first")
        dt_ny = dt_naive.dt.tz_localize(
            "America/New_York", ambiguous=(~is_dup).to_numpy(), nonexistent="raise"
        )
        g["ts_utc"] = dt_ny.dt.tz_convert("UTC").dt.tz_localize(None)
        out.append(g[["Name", "ts_utc", "LBMP ($/MWHr)"]])
    return pd.concat(out, ignore_index=True)


def verify_utc_alignment(cems: pd.DataFrame, zone_utc: pd.DataFrame) -> dict:
    """Verify exact UTC/DST hour alignment via lag correlations and startup hour modes."""
    z_nyc = zone_utc[zone_utc["Name"] == "N.Y.C."].sort_values("ts_utc")
    z_li = zone_utc[zone_utc["Name"] == "LONGIL"].sort_values("ts_utc")
    c_tot = cems.groupby("operating_datetime_utc")["gross_load_mw"].sum().rename("mw")
    m = pd.merge(z_nyc, c_tot.reset_index(), left_on="ts_utc", right_on="operating_datetime_utc").sort_values("ts_utc")
    m["mo"] = m["ts_utc"].dt.month

    c_li = cems[cems["plant_id_eia"].isin([2511, 2516])].groupby("operating_datetime_utc")["gross_load_mw"].sum().rename("mw")
    mli = pd.merge(z_li, c_li.reset_index(), left_on="ts_utc", right_on="operating_datetime_utc").sort_values("ts_utc")
    mli["mo"] = mli["ts_utc"].dt.month

    lags = [-2, -1, 0, 1, 2]
    def lag_dict(sub, diff=False):
        x = sub["mw"].diff() if diff else sub["mw"]
        y = sub["LBMP ($/MWHr)"].diff() if diff else sub["LBMP ($/MWHr)"]
        return {f"{k:+d}h": float(x.corr(y.shift(k))) for k in lags}

    # Check morning startup jump in local NY time for Winter (EST) vs Summer (EDT)
    cs = cems.sort_values(["plant_id_eia", "emissions_unit_id_epa", "operating_datetime_utc"])
    prev_op = cs.groupby(["plant_id_eia", "emissions_unit_id_epa"])["operating_time_hours"].shift(1)
    starts = cs[(cs["operating_time_hours"] > 0) & (prev_op == 0)].copy()
    ts_ny = starts["operating_datetime_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    starts["hr_ny"] = ts_ny.dt.hour
    starts["mo"] = ts_ny.dt.month
    win_h = starts[starts["mo"].isin([1, 2])]["hr_ny"].value_counts()
    sum_h = starts[starts["mo"].isin([7, 8])]["hr_ny"].value_counts()

    return {
        "n_merged_hours": int(len(m)),
        "utc_start": str(m["ts_utc"].min()),
        "utc_end": str(m["ts_utc"].max()),
        "all_ny_level_corr_vs_nyc_lbmp": {
            "full_year": lag_dict(m, diff=False),
            "winter_jan_feb_est": lag_dict(m[m["mo"].isin([1, 2])], diff=False),
            "summer_jul_aug_edt": lag_dict(m[m["mo"].isin([7, 8])], diff=False),
        },
        "longil_steam_diff_corr_vs_longil_lbmp": {
            "winter_jan_feb_est": lag_dict(mli[mli["mo"].isin([1, 2])], diff=True),
            "spring_apr_may_edt": lag_dict(mli[mli["mo"].isin([4, 5])], diff=True),
        },
        "morning_startup_jump_local_ny": {
            "winter_est_h05_to_h06": [int(win_h.get(5, 0)), int(win_h.get(6, 0))],
            "summer_edt_h05_to_h06": [int(sum_h.get(5, 0)), int(sum_h.get(6, 0))],
        },
    }


# ---------------------------------------------------------------------------
# Step 2: Fast exact single-unit HiGHS LP & QP solvers
# ---------------------------------------------------------------------------

def solve_kink_lp(
    margin: np.ndarray,
    pmin: np.ndarray,
    rmax: np.ndarray,
    ramp: float,
    consec: np.ndarray,
    sigma: float = 0.85,
    w_grid: np.ndarray = W_GRID,
) -> np.ndarray:
    """Exact single-unit wear-kink LP dispatch across w_grid via warm-started HiGHS."""
    T = len(margin)
    free = sigma * rmax
    s_lo = np.minimum(pmin, free)
    s_hi = np.maximum(s_lo, free)
    e_lo = np.maximum(0.0, pmin - free)
    e_hi = np.maximum(e_lo, rmax - np.maximum(free, pmin))
    idx_c = np.where(consec)[0]
    nr = len(idx_c)
    nv = 2 * T
    if nr > 0:
        r_k = np.repeat(np.arange(nr, dtype=np.int32), 4)
        c_j = np.empty(4 * nr, dtype=np.int32)
        c_j[0::4] = idx_c
        c_j[1::4] = T + idx_c
        c_j[2::4] = idx_c + 1
        c_j[3::4] = T + idx_c + 1
        M = csc_matrix((np.tile([-1.0, -1.0, 1.0, 1.0], nr), (r_k, c_j)), shape=(nr, nv))
    else:
        M = csc_matrix((0, nv))

    lp = _h.HighsLp()
    lp.num_col_, lp.num_row_ = nv, nr
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = nv, nr
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = M.indptr, M.indices, M.data
    lp.col_lower_ = np.concatenate([s_lo, e_lo])
    lp.col_upper_ = np.concatenate([s_hi, e_hi])
    lp.col_cost_ = np.concatenate([-margin, 1e-7 - margin])
    lp.row_lower_ = np.full(nr, -ramp)
    lp.row_upper_ = np.full(nr, ramp)
    h = _h._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    e_idx = np.arange(T, 2 * T, dtype=np.int32)

    P = np.empty((len(w_grid), T), dtype=float)
    saturated = False
    p_sat = None
    for k, w in enumerate(w_grid):
        if saturated:
            P[k] = p_sat
            continue
        h.changeColsCost(T, e_idx, max(float(w), 1e-7) - margin)
        h.run()
        x = np.asarray(h.getSolution().col_value)
        p = x[:T] + x[T:]
        P[k] = p
        # Exact optimality short-circuit: once all e_t sit at e_lo, larger w cannot change solution.
        if np.all(x[T:] <= e_lo + 1e-6):
            saturated = True
            p_sat = p
    return P


def solve_quad_qp(
    margin: np.ndarray,
    pmin: np.ndarray,
    rmax: np.ndarray,
    ramp: float,
    consec: np.ndarray,
    g_grid: np.ndarray = G_GRID,
) -> np.ndarray:
    """Exact single-unit smooth quadratic cost QP via HiGHS.

    Marginal cost: c_marg(p) = c_t + gamma * (p - pmin) / (rmax - pmin), 1 extra DOF.
    """
    T = len(margin)
    idx_c = np.where(consec)[0]
    nr = len(idx_c)
    if nr > 0:
        r_k = np.repeat(np.arange(nr, dtype=np.int32), 2)
        c_j = np.empty(2 * nr, dtype=np.int32)
        c_j[0::2] = idx_c
        c_j[1::2] = idx_c + 1
        M = csc_matrix((np.tile([-1.0, 1.0], nr), (r_k, c_j)), shape=(nr, T))
    else:
        M = csc_matrix((0, T))

    lp = _h.HighsLp()
    lp.num_col_, lp.num_row_ = T, nr
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = T, nr
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = M.indptr, M.indices, M.data
    lp.col_lower_ = pmin
    lp.col_upper_ = rmax
    lp.col_cost_ = -margin
    lp.row_lower_ = np.full(nr, -ramp)
    lp.row_upper_ = np.full(nr, ramp)
    h = _h._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    all_idx = np.arange(T, dtype=np.int32)
    span = np.maximum(rmax - pmin, 1.0)
    h_start = np.arange(T + 1, dtype=np.int32)
    h_index = np.arange(T, dtype=np.int32)

    P = np.empty((len(g_grid), T), dtype=float)
    for k, gm in enumerate(g_grid):
        if gm <= 0.0:
            h.changeColsCost(T, all_idx, -margin)
            h.run()
            P[k] = np.asarray(h.getSolution().col_value)
        else:
            q = float(gm) / span
            hess = _h.HighsHessian()
            hess.dim_ = T
            hess.format_ = _h.HessianFormat.kTriangular
            hess.start_ = h_start
            hess.index_ = h_index
            hess.value_ = q
            h.passHessian(hess)
            h.changeColsCost(T, all_idx, -margin - q * pmin)
            h.run()
            P[k] = np.asarray(h.getSolution().col_value)
    return P


def solve_cap_lp(
    margin: np.ndarray,
    pmin: np.ndarray,
    rmax: np.ndarray,
    ramp: float,
    consec: np.ndarray,
    th_grid: np.ndarray = TH_GRID,
) -> np.ndarray:
    """Exact single-unit blind LP (w=0) with 1-param constant operational ceiling theta * rmax."""
    T = len(margin)
    idx_c = np.where(consec)[0]
    nr = len(idx_c)
    if nr > 0:
        r_k = np.repeat(np.arange(nr, dtype=np.int32), 2)
        c_j = np.empty(2 * nr, dtype=np.int32)
        c_j[0::2] = idx_c
        c_j[1::2] = idx_c + 1
        M = csc_matrix((np.tile([-1.0, 1.0], nr), (r_k, c_j)), shape=(nr, T))
    else:
        M = csc_matrix((0, T))

    lp = _h.HighsLp()
    lp.num_col_, lp.num_row_ = T, nr
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = T, nr
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = M.indptr, M.indices, M.data
    lp.col_lower_ = pmin
    lp.col_upper_ = rmax
    lp.col_cost_ = -margin
    lp.row_lower_ = np.full(nr, -ramp)
    lp.row_upper_ = np.full(nr, ramp)
    h = _h._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    all_idx = np.arange(T, dtype=np.int32)

    P = np.empty((len(th_grid), T), dtype=float)
    for k, th in enumerate(th_grid):
        ub = np.maximum(pmin, float(th) * rmax)
        h.changeColsBounds(T, all_idx, pmin, ub)
        h.run()
        P[k] = np.asarray(h.getSolution().col_value)
    return P


def estimate_profile(P: np.ndarray, p_obs: np.ndarray, w_grid: np.ndarray = W_GRID, tol: float = TOL) -> dict:
    """Extract point estimate, 5%-SSE profile interval [w_lo, w_hi], and SSE metrics."""
    sse = np.sum((P - p_obs[None, :]) ** 2, axis=1)
    j = int(np.argmin(sse))
    smin = float(sse[j])
    s0 = float(sse[0])
    thr = smin * tol + 1e-6
    jl, jr = j, j
    while jl > 0 and sse[jl - 1] <= thr:
        jl -= 1
    while jr < len(sse) - 1 and sse[jr + 1] <= thr:
        jr += 1
    w_hat = float(w_grid[j])
    w_lo = float(w_grid[jl])
    w_hi = float(w_grid[jr])
    return {
        "w_hat": w_hat,
        "w_lo": w_lo,
        "w_hi": w_hi,
        "ident": bool(w_hi < w_grid[-1] - 1e-6),
        "excl0": bool(w_lo > 0.0),
        "two_sided": bool((w_lo > 0.0) and (w_hi < w_grid[-1] - 1e-6)),
        "sse0": s0,
        "sse_min": smin,
        "sse_imp": float((s0 - smin) / max(s0, 1e-12)),
        "j_hat": j,
    }


def verify_solvers() -> dict:
    """Verify LP and QP solvers against exact zero-ramp analytical limits."""
    rng = np.random.default_rng(12345)
    T = 168
    margin = rng.normal(12.0, 28.0, T)
    pmin = np.full(T, 90.0)
    rmax = np.full(T, 310.0)
    sigma = 0.85
    consec_none = np.zeros(T - 1, dtype=bool)
    w_test = np.array([0.0, 15.0, 30.0, 60.0])
    P_lp = solve_kink_lp(margin, pmin, rmax, ramp=1e6, consec=consec_none, sigma=sigma, w_grid=w_test)
    diffs_lp = []
    for k, w in enumerate(w_test):
        p_ana = np.where(margin < 0.0, pmin, np.where(margin < w, np.maximum(pmin, sigma * rmax), rmax))
        diffs_lp.append(float(np.max(np.abs(P_lp[k] - p_ana))))

    g_test = np.array([0.0, 20.0, 50.0])
    P_qp = solve_quad_qp(margin, pmin, rmax, ramp=1e6, consec=consec_none, g_grid=g_test)
    diffs_qp = []
    for k, g in enumerate(g_test):
        p_ana = np.where(margin >= 0.0, rmax, pmin) if g == 0.0 else np.clip(pmin + (rmax - pmin) * margin / g, pmin, rmax)
        diffs_qp.append(float(np.max(np.abs(P_qp[k] - p_ana))))
    return {
        "zero_ramp_lp_max_abs_diff_mw": float(max(diffs_lp)),
        "zero_ramp_qp_max_abs_diff_mw": float(max(diffs_qp)),
    }


# ---------------------------------------------------------------------------
# Cohort construction (strictly outcome-independent)
# ---------------------------------------------------------------------------

def build_unit_dataset() -> tuple[list[dict], list[dict], dict, dict]:
    """Load cached NY datasets, verify UTC alignment, and apply outcome-independent filters."""
    ensure_data_cached()
    cems = pd.read_parquet(os.path.join(DATA_DIR, "cems_ny_2024.parquet"))
    plants = pd.read_parquet(os.path.join(DATA_DIR, "eia_entity_plants_ny.parquet"))
    scd = pd.read_parquet(os.path.join(DATA_DIR, "eia860_scd_plants_ny.parquet"))
    fuel = pd.read_parquet(os.path.join(DATA_DIR, "eia923_fuel_ny.parquet"))
    zone = pd.read_parquet(os.path.join(DATA_DIR, "nyiso_damlbmp_zone_2024.parquet"))
    ngen = pd.read_parquet(os.path.join(DATA_DIR, "nyiso_damlbmp_gen_2024.parquet"))

    n_raw_units = int(cems.groupby(["plant_id_eia", "emissions_unit_id_epa"]).ngroups)

    # Consolidate twin boilers mapping to same EIA generator with identical gross_load_mw
    is_twin = cems.apply(lambda r: (r["plant_id_eia"], r["emissions_unit_id_epa"]) in TWIN_BOILER_MAP, axis=1)
    cems_nt = cems[~is_twin]
    cems_tw = cems[is_twin].copy()
    cems_tw["emissions_unit_id_epa"] = [
        TWIN_BOILER_MAP[(p, u)] for p, u in zip(cems_tw["plant_id_eia"], cems_tw["emissions_unit_id_epa"])
    ]
    cems_tw = cems_tw.groupby(
        ["state", "year", "plant_id_eia", "emissions_unit_id_epa", "operating_datetime_utc"], as_index=False
    ).agg({
        "plant_id_epa": "first",
        "operating_time_hours": "max",
        "gross_load_mw": "max",
        "steam_load_lbs": "sum",
        "heat_content_mmbtu": "sum",
        "co2_mass_tons": "sum",
        "so2_mass_lbs": "sum",
        "nox_mass_lbs": "sum",
    })
    cems_all = pd.concat([cems_nt, cems_tw], ignore_index=True)
    n_after_twin = int(cems_all.groupby(["plant_id_eia", "emissions_unit_id_epa"]).ngroups)

    # Parse NYISO zone and needed generator nodes to UTC
    needed_zones = sorted({v[1] for v in NYISO_NODE_MAP.values()})
    zone_utc = parse_nyiso_to_utc(zone[zone["Name"].isin(needed_zones)])
    needed_nodes = sorted({v[0] for v in NYISO_NODE_MAP.values()})
    ngen_utc = parse_nyiso_to_utc(ngen[ngen["Name"].isin(needed_nodes)])

    utc_audit = verify_utc_alignment(cems, zone_utc)

    # EIA-923 2024 monthly delivered gas and oil prices
    fuel["year"] = pd.to_datetime(fuel["report_date"]).dt.year
    fuel["month"] = pd.to_datetime(fuel["report_date"]).dt.month
    f24_gas = fuel[(fuel["year"] == 2024) & (fuel["fuel_type_code_pudl"] == "gas") & fuel["fuel_cost_per_mmbtu"].notna()]
    f24_oil = fuel[(fuel["year"] == 2024) & (fuel["fuel_type_code_pudl"] == "oil") & fuel["fuel_cost_per_mmbtu"].notna()]
    ny_mo_gas = f24_gas.groupby("month")["fuel_cost_per_mmbtu"].median().to_dict()
    ny_oil_price = float(f24_oil["fuel_cost_per_mmbtu"].median())

    latest_scd = scd.sort_values("report_date").groupby("plant_id_eia").last()
    plants_idx = plants.set_index("plant_id_eia")

    # Apply outcome-independent cohort filter funnel
    funnel = {
        "raw_ny_2024_cems_units": n_raw_units,
        "after_twin_boiler_consolidation": n_after_twin,
        "scale_ge100mw_and_ge2000h": 0,
        "exact_nyiso_node_and_ge10mo_eia923_fuel": 0,
        "non_cogen_eia860": 0,
        "clean_heat_rate_r2_ge090": 0,
        "dispatch_variability_std_ge008_primary_cohort": 0,
        "extended_cohort_including_cogen_ccgts": 0,
        "ny_2024_median_gas_usd_mmbtu_by_month": {int(k): float(v) for k, v in ny_mo_gas.items()},
        "ny_2024_median_oil_usd_mmbtu": ny_oil_price,
    }

    primary_units = []
    extended_units = []

    for (pid, uid), g in sorted(cems_all.groupby(["plant_id_eia", "emissions_unit_id_epa"])):
        pid_i, uid_s = int(pid), str(uid)
        full = g[(g["operating_time_hours"] == 1.0) & (g["gross_load_mw"] > 0) & (g["heat_content_mmbtu"] > 0)].copy()
        if len(full) < 2000 or full["gross_load_mw"].quantile(0.99) < 100.0:
            continue
        funnel["scale_ge100mw_and_ge2000h"] += 1

        # Exact NYISO node match + >= 10 months EIA-923 2024 gas cost
        n_fmo = int(f24_gas[f24_gas["plant_id_eia"] == pid_i]["month"].nunique())
        if (pid_i, uid_s) not in NYISO_NODE_MAP or n_fmo < 10:
            continue
        funnel["exact_nyiso_node_and_ge10mo_eia923_fuel"] += 1

        sec = str(latest_scd.loc[pid_i, "sector_name_eia"]) if pid_i in latest_scd.index else ""
        is_cogen = (("Cogen" in sec) and ("Non-Cogen" not in sec)) or (("CHP" in sec) and ("Non-CHP" not in sec))
        if not is_cogen:
            funnel["non_cogen_eia860"] += 1

        # Remove low-load startup/sync artifacts (<10% of P99) before estimating heat curve & dispatch
        p99_raw = float(full["gross_load_mw"].quantile(0.99))
        full = full[full["gross_load_mw"] >= 0.10 * p99_raw].copy()
        hr = full["heat_content_mmbtu"] / full["gross_load_mw"]
        v = full[(hr >= hr.quantile(0.005)) & (hr <= hr.quantile(0.995))]
        x_v = v["gross_load_mw"].to_numpy(float)
        y_v = v["heat_content_mmbtu"].to_numpy(float)
        b_lin, a_lin = np.polyfit(x_v, y_v, 1)
        ss_tot = float(np.sum((y_v - y_v.mean()) ** 2))
        r2_lin = float(1.0 - np.sum((y_v - (a_lin + b_lin * x_v)) ** 2) / max(ss_tot, 1e-12))
        c2, c1, c0 = np.polyfit(x_v, y_v, 2)
        r2_quad = float(1.0 - np.sum((y_v - (c0 + c1 * x_v + c2 * x_v ** 2)) ** 2) / max(ss_tot, 1e-12))

        if r2_lin < 0.90:
            continue
        if not is_cogen:
            funnel["clean_heat_rate_r2_ge090"] += 1

        r_ann = float(full["gross_load_mw"].quantile(0.995))
        std_r = float(full["gross_load_mw"].std() / r_ann)
        if std_r < 0.08:
            continue

        gname, zname, tech = NYISO_NODE_MAP[(pid_i, uid_s)]
        gn = ngen_utc[ngen_utc["Name"] == gname][["ts_utc", "LBMP ($/MWHr)"]].rename(columns={"LBMP ($/MWHr)": "lbmp_node"})
        zn = zone_utc[zone_utc["Name"] == zname][["ts_utc", "LBMP ($/MWHr)"]].rename(columns={"LBMP ($/MWHr)": "lbmp_zone"})
        m = pd.merge(full, gn, left_on="operating_datetime_utc", right_on="ts_utc", how="inner")
        m = pd.merge(m, zn, on="ts_utc", how="inner").sort_values("ts_utc").reset_index(drop=True)
        m["mo"] = m["ts_utc"].dt.month
        m["season"] = m["mo"].map({
            12: "DJF", 1: "DJF", 2: "DJF",
            3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA",
            9: "SON", 10: "SON", 11: "SON",
        })

        co2_rate = float((v["co2_mass_tons"] / v["gross_load_mw"]).median())
        fp = f24_gas[f24_gas["plant_id_eia"] == pid_i].groupby("month")["fuel_cost_per_mmbtu"].mean().to_dict()
        gas_p = m["mo"].map(lambda k: fp.get(k, ny_mo_gas[k])).to_numpy(float)
        co2_per_mmbtu = (m["co2_mass_tons"] / m["heat_content_mmbtu"]).to_numpy(float)
        is_oil = co2_per_mmbtu > 0.068

        c_gas = b_lin * gas_p + co2_rate * RGGI_USD_PER_SHORT_TON
        co2_hr = np.clip((m["co2_mass_tons"] / m["gross_load_mw"]).to_numpy(float), 0.2, 1.5)
        c_oil_adj = b_lin * np.where(is_oil, ny_oil_price, gas_p) + co2_hr * RGGI_USD_PER_SHORT_TON

        # Seasonal P99.5, Monthly P99, and Annual P99.5 capacities
        r_s_map = {}
        for s_name, grp in m.groupby("season"):
            r_s_map[s_name] = float(grp["gross_load_mw"].quantile(0.995)) if len(grp) >= 100 else r_ann
        r_m_map = {}
        for mo_k, grp in m.groupby("mo"):
            r_m_map[mo_k] = float(grp["gross_load_mw"].quantile(0.99)) if len(grp) >= 100 else r_s_map[grp["season"].iloc[0]]

        r_seas = m["season"].map(r_s_map).to_numpy(float)
        r_mon = m["mo"].map(r_m_map).to_numpy(float)
        pmin_val = float(m["gross_load_mw"].quantile(0.05))
        ts_ns = m["ts_utc"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        consec_all = (np.diff(ts_ns) == 3600)
        p_mw = m["gross_load_mw"].to_numpy(float)
        ramp_p99 = float(np.quantile(np.abs(np.diff(p_mw))[consec_all], 0.99))

        # Duct-firing / incremental heat rate below vs above 0.85 * r_seas
        v_lf = v["gross_load_mw"].to_numpy(float) / v["operating_datetime_utc"].dt.month.map(
            {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
        ).map(r_s_map).to_numpy(float)
        lo_m = (v_lf >= 0.50) & (v_lf <= 0.85)
        hi_m = (v_lf > 0.85) & (v_lf <= 1.00)
        if lo_m.sum() >= 100 and hi_m.sum() >= 100 and x_v[hi_m].std() > 3.0:
            b_lo85 = float(np.polyfit(x_v[lo_m], y_v[lo_m], 1)[0])
            b_hi85 = float(np.polyfit(x_v[hi_m], y_v[hi_m], 1)[0])
            delta_b = b_hi85 - b_lo85
        else:
            b_lo85, b_hi85, delta_b = None, None, None

        lbmp_node = m["lbmp_node"].to_numpy(float)
        lbmp_zone = m["lbmp_zone"].to_numpy(float)
        pname = str(plants_idx.loc[pid_i, "plant_name_eia"]) if pid_i in plants_idx.index else str(pid_i)
        county = str(plants_idx.loc[pid_i, "county"]) if pid_i in plants_idx.index else ""
        r_sum = r_s_map.get("JJA")
        r_win = r_s_map.get("DJF")
        sw_ratio = float(r_sum / r_win) if (r_sum and r_win and "JJA" in m["season"].values and "DJF" in m["season"].values) else None

        unit_obj = {
            "meta": {
                "unit_key": f"{pid_i}:{uid_s}",
                "plant_id_eia": pid_i,
                "unit_id": uid_s,
                "plant_name": pname,
                "county": county,
                "sector_eia860": sec,
                "is_cogen": bool(is_cogen),
                "tech": tech,
                "nyiso_node": gname,
                "nyiso_zone": zname,
                "n_hours": int(len(m)),
                "oil_hours": int(is_oil.sum()),
                "b_linear_mmbtu_mwh": float(b_lin),
                "a_linear_mmbtu": float(a_lin),
                "r2_linear": r2_lin,
                "r2_quadratic": r2_quad,
                "b_below_85": b_lo85,
                "b_above_85": b_hi85,
                "delta_b_above_85": delta_b,
                "co2_rate_ton_mwh": co2_rate,
                "pmin_mw": pmin_val,
                "r_annual_mw": r_ann,
                "r_summer_mw": float(r_sum) if r_sum is not None else None,
                "r_winter_mw": float(r_win) if r_win is not None else None,
                "summer_winter_cap_ratio": sw_ratio,
                "ramp_p99_mw_h": ramp_p99,
                "std_over_r_ann": std_r,
                "mean_lbmp_node": float(lbmp_node.mean()),
                "mean_c_marg": float(c_gas.mean()),
                "nodal_vs_zonal_mae": float(np.mean(np.abs(lbmp_node - lbmp_zone))),
                "nodal_vs_zonal_corr": float(np.corrcoef(lbmp_node, lbmp_zone)[0, 1]),
            },
            "arrays": {
                "ts_sec": ts_ns,
                "month": m["mo"].to_numpy(np.int32),
                "season": m["season"].to_numpy(str),
                "p_mw": p_mw,
                "lbmp_node": lbmp_node,
                "lbmp_zone": lbmp_zone,
                "c_gas": c_gas,
                "c_oil_adj": c_oil_adj,
                "r_seas": r_seas,
                "r_mon": r_mon,
                "r_ann": np.full(len(m), r_ann, dtype=float),
                "pmin_val": pmin_val,
                "ramp": ramp_p99,
            },
        }
        extended_units.append(unit_obj)
        if not is_cogen:
            primary_units.append(unit_obj)

    funnel["dispatch_variability_std_ge008_primary_cohort"] = len(primary_units)
    funnel["extended_cohort_including_cogen_ccgts"] = len(extended_units)
    return primary_units, extended_units, funnel, utc_audit


# ---------------------------------------------------------------------------
# Non-parametric margin-response & bunching diagnostics
# ---------------------------------------------------------------------------

def compute_margin_response(units: list[dict]) -> dict:
    """Compute binned loading p_t / r_t vs net margin m_t = lbmp - c_{g,t}."""
    bins = [-50.0, -5.0, 0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 70.0, 500.0]
    rows = []
    for u in units:
        a = u["arrays"]
        tech = u["meta"]["tech"]
        margin = a["lbmp_node"] - a["c_gas"]
        margin_adj = a["lbmp_node"] - a["c_oil_adj"]
        rows.append(pd.DataFrame({
            "tech": tech,
            "mo": a["month"],
            "lf_s": np.clip(a["p_mw"] / a["r_seas"], 0.0, 1.05),
            "lf_m": np.clip(a["p_mw"] / a["r_mon"], 0.0, 1.05),
            "lf_a": np.clip(a["p_mw"] / a["r_ann"], 0.0, 1.05),
            "margin": margin,
            "margin_adj": margin_adj,
        }))
    df = pd.concat(rows, ignore_index=True)

    def bin_table(sub: pd.DataFrame, lf_col: str = "lf_s", m_col: str = "margin") -> list[dict]:
        out = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            s = sub[(sub[m_col] >= b0) & (sub[m_col] < b1)]
            if len(s) < 20:
                continue
            v = s[lf_col].to_numpy(float)
            out.append({
                "m_lo": b0,
                "m_hi": b1,
                "m_mid": float(np.median(s[m_col])),
                "n": int(len(v)),
                "mean": float(np.mean(v)),
                "q25": float(np.quantile(v, 0.25)),
                "median": float(np.median(v)),
                "q75": float(np.quantile(v, 0.75)),
                "frac_in_080_090": float(np.mean((v >= 0.80) & (v <= 0.90))),
                "frac_gt_090": float(np.mean(v > 0.90)),
            })
        return out

    nw = ~df["mo"].isin([1, 2, 12])
    win = df["mo"].isin([1, 2, 12])
    return {
        "n_unit_hours_total": int(len(df)),
        "n_unit_hours_mar_nov": int(nw.sum()),
        "all_full_year_seas": bin_table(df, "lf_s"),
        "all_mar_nov_seas": bin_table(df[nw], "lf_s"),
        "all_winter_djf_seas": bin_table(df[win], "lf_s"),
        "all_full_year_annual_cap": bin_table(df, "lf_a"),
        "all_full_year_oil_adj": bin_table(df, "lf_s", "margin_adj"),
        "ccgt_full_year_seas": bin_table(df[df["tech"] == "CCGT"], "lf_s"),
        "ccgt_mar_nov_seas": bin_table(df[(df["tech"] == "CCGT") & nw], "lf_s"),
        "ccgt_mar_nov_monthly_cap": bin_table(df[(df["tech"] == "CCGT") & nw], "lf_m"),
        "ccgt_mar_nov_annual_cap": bin_table(df[(df["tech"] == "CCGT") & nw], "lf_a"),
        "st_full_year_seas": bin_table(df[df["tech"] == "ST"], "lf_s"),
        "st_mar_nov_seas": bin_table(df[(df["tech"] == "ST") & nw], "lf_s"),
    }


# ---------------------------------------------------------------------------
# Parallel workers for window & full-year estimation
# ---------------------------------------------------------------------------

def _block_bootstrap(resid: np.ndarray, blk: int, rng: np.random.Generator) -> np.ndarray:
    T = len(resid)
    nb = (T + blk - 1) // blk
    starts = rng.integers(0, max(1, T - blk + 1), size=nb)
    e = np.concatenate([resid[st:st + blk] for st in starts])[:T]
    return e - float(np.mean(e))


def _window_slices(ts_sec: np.ndarray, win_h: int, min_frac: float = 0.80) -> list[np.ndarray]:
    """Non-overlapping calendar windows of length win_h hours starting 2024-01-01 05:00 UTC."""
    t0 = int(np.datetime64("2024-01-01T05:00:00", "s").astype(np.int64))
    step = win_h * 3600
    min_n = int(min_frac * win_h)
    slices = []
    for k in range(8784 // win_h):
        lo = t0 + k * step
        hi = lo + step
        idx = np.where((ts_sec >= lo) & (ts_sec < hi))[0]
        if len(idx) >= min_n:
            slices.append(idx)
    return slices


def _worker_unit_windows(args: tuple) -> list[dict]:
    """Process one unit's windows for a given specification."""
    unit_meta, arr, cfg = args
    win_h = cfg["win_h"]
    sigma = cfg["sigma"]
    cap_mode = cfg["cap_mode"]
    lbmp_mode = cfg["lbmp_mode"]
    cost_mode = cfg["cost_mode"]
    cost_shift = cfg["cost_shift"]
    do_placebos = cfg["do_placebos"]
    do_horse_race = cfg["do_horse_race"]
    seed_base = cfg["seed_base"]

    ts_sec = arr["ts_sec"]
    p_all = arr["p_mw"]
    lbmp_all = arr["lbmp_node"] if lbmp_mode == "nodal" else arr["lbmp_zone"]
    c_all = (arr["c_oil_adj"] if cost_mode == "oil_adj" else arr["c_gas"]) + cost_shift
    rmax_all = arr["r_ann"] if cap_mode == "annual" else arr["r_seas"]
    pmin_val = arr["pmin_val"]
    ramp = arr["ramp"]
    mo_all = arr["month"]
    seas_all = arr["season"]

    rng = np.random.default_rng(seed_base + unit_meta["plant_id_eia"] * 131 + sum(ord(c) for c in unit_meta["unit_id"]))
    slices = _window_slices(ts_sec, win_h, min_frac=0.80)
    recs = []

    for w_idx, idx in enumerate(slices):
        p_obs = p_all[idx]
        lbmp = lbmp_all[idx]
        cost = c_all[idx]
        margin = lbmp - cost
        rmax = rmax_all[idx]
        pmin = np.full(len(idx), pmin_val, dtype=float)
        consec = (np.diff(ts_sec[idx]) == 3600)
        mo = int(mo_all[idx[len(idx) // 2]])
        seas = str(seas_all[idx[len(idx) // 2]])

        P = solve_kink_lp(margin, pmin, rmax, ramp, consec, sigma=sigma, w_grid=W_GRID)
        est_r = estimate_profile(P, p_obs, W_GRID, TOL)

        rec = {
            "unit": unit_meta["unit_key"],
            "tech": unit_meta["tech"],
            "zone": unit_meta["nyiso_zone"],
            "month": mo,
            "season": seas,
            "win_idx": w_idx,
            "T": int(len(idx)),
            "max_margin": float(np.max(margin)),
            "real": {k: v for k, v in est_r.items() if k != "j_hat"},
        }

        if do_placebos:
            # 1. LBMP-shuffle within window
            lbmp_sh = lbmp[rng.permutation(len(idx))]
            P_sh = solve_kink_lp(lbmp_sh - cost, pmin, rmax, ramp, consec, sigma=sigma, w_grid=W_GRID)
            est_sh = estimate_profile(P_sh, p_obs, W_GRID, TOL)
            rec["lbmp_shuffled"] = {k: v for k, v in est_sh.items() if k != "j_hat"}

            # Empirical residuals at fitted w_hat
            resid = p_obs - P[est_r["j_hat"]]
            e_blk = _block_bootstrap(resid, blk=24, rng=rng)
            e_perm = resid[rng.permutation(len(idx))] - float(np.mean(resid))
            e_g5 = rng.normal(0.0, 0.05 * float(np.mean(rmax)), size=len(idx))

            # 2. Null placebos (w_true = 0)
            rec["null_blk24"] = {k: v for k, v in estimate_profile(P, P[0] + e_blk, W_GRID, TOL).items() if k != "j_hat"}
            rec["null_iid"] = {k: v for k, v in estimate_profile(P, P[0] + e_perm, W_GRID, TOL).items() if k != "j_hat"}
            rec["null_gauss5pct"] = {k: v for k, v in estimate_profile(P, P[0] + e_g5, W_GRID, TOL).items() if k != "j_hat"}

            # 3. Positive controls w_true in {15, 30, 60, 120} with 24-h block-bootstrapped real residuals
            pos_out = {}
            for wt in POS_W_LIST:
                jwt = int(np.where(W_GRID == wt)[0][0])
                ep = estimate_profile(P, P[jwt] + e_blk, W_GRID, TOL)
                pos_out[str(int(wt))] = {k: v for k, v in ep.items() if k != "j_hat"}
            rec["pos_blk24"] = pos_out

            if do_horse_race:
                # 4. Equal-DOF (1 extra param) model comparison: Kink LP vs Smooth Quad QP vs Const-Cap LP
                Pq = solve_quad_qp(margin, pmin, rmax, ramp, consec, g_grid=G_GRID)
                Pc = solve_cap_lp(margin, pmin, rmax, ramp, consec, th_grid=TH_GRID)
                sse_q = np.sum((Pq - p_obs[None, :]) ** 2, axis=1)
                sse_c = np.sum((Pc - p_obs[None, :]) ** 2, axis=1)
                jq, jc = int(np.argmin(sse_q)), int(np.argmin(sse_c))
                rec["horse_race_real"] = {
                    "sse0": est_r["sse0"],
                    "sse_kink": est_r["sse_min"],
                    "sse_quad": float(sse_q[jq]),
                    "sse_cap": float(sse_c[jc]),
                    "w_hat": est_r["w_hat"],
                    "gamma_hat": float(G_GRID[jq]),
                    "theta_hat": float(TH_GRID[jc]),
                }
                # Discriminative-power sanity test: exact same 3-model race on planted w_true = 30!
                j30 = int(np.where(W_GRID == 30.0)[0][0])
                y_pos30 = P[j30] + e_blk
                sse_k30 = float(np.min(np.sum((P - y_pos30[None, :]) ** 2, axis=1)))
                sse_q30 = float(np.min(np.sum((Pq - y_pos30[None, :]) ** 2, axis=1)))
                sse_c30 = float(np.min(np.sum((Pc - y_pos30[None, :]) ** 2, axis=1)))
                rec["horse_race_pos30"] = {
                    "sse_kink": sse_k30,
                    "sse_quad": sse_q30,
                    "sse_cap": sse_c30,
                }
        recs.append(rec)
    return recs


def _worker_unit_fullyear(args: tuple) -> dict:
    """Full-year and non-winter (Mar-Nov) unit-level LP estimation + controls."""
    unit_meta, arr = args
    ts_sec = arr["ts_sec"]
    p_all = arr["p_mw"]
    lbmp = arr["lbmp_node"]
    c_gas = arr["c_gas"]
    c_oil = arr["c_oil_adj"]
    r_seas = arr["r_seas"]
    r_ann = arr["r_ann"]
    pmin = np.full(len(p_all), arr["pmin_val"], dtype=float)
    ramp = arr["ramp"]
    consec = (np.diff(ts_sec) == 3600)
    mo = arr["month"]

    # 1. Full year, seasonal cap
    P_s = solve_kink_lp(lbmp - c_gas, pmin, r_seas, ramp, consec, sigma=0.85, w_grid=W_GRID)
    est_s = estimate_profile(P_s, p_all, W_GRID, TOL)

    # 2. Full year, annual cap (ambient derating confounder)
    P_a = solve_kink_lp(lbmp - c_gas, pmin, r_ann, ramp, consec, sigma=0.85, w_grid=W_GRID)
    est_a = estimate_profile(P_a, p_all, W_GRID, TOL)

    # 3. Full year, oil-switch adjusted cost
    P_oil = solve_kink_lp(lbmp - c_oil, pmin, r_seas, ramp, consec, sigma=0.85, w_grid=W_GRID)
    est_oil = estimate_profile(P_oil, p_all, W_GRID, TOL)

    # 4. Non-winter Mar-Nov, seasonal cap
    nw = ~np.isin(mo, [1, 2, 12])
    ts_nw = ts_sec[nw]
    P_nw = solve_kink_lp((lbmp - c_gas)[nw], pmin[nw], r_seas[nw], ramp, np.diff(ts_nw) == 3600, sigma=0.85, w_grid=W_GRID)
    est_nw = estimate_profile(P_nw, p_all[nw], W_GRID, TOL)

    # 5. Positive control (w=15, 30, 60) & null (w=0) on full year with 24-h block-bootstrapped residuals
    rng = np.random.default_rng(77000 + unit_meta["plant_id_eia"] * 37 + sum(ord(c) for c in unit_meta["unit_id"]))
    resid = p_all - P_s[est_s["j_hat"]]
    e_blk = _block_bootstrap(resid, blk=24, rng=rng)
    null0 = estimate_profile(P_s, P_s[0] + e_blk, W_GRID, TOL)
    pos = {}
    for wt in (15.0, 30.0, 60.0):
        jwt = int(np.where(W_GRID == wt)[0][0])
        ep = estimate_profile(P_s, P_s[jwt] + e_blk, W_GRID, TOL)
        pos[str(int(wt))] = {k: v for k, v in ep.items() if k != "j_hat"}

    clean = lambda d: {k: v for k, v in d.items() if k != "j_hat"}
    return {
        "unit_key": unit_meta["unit_key"],
        "plant_name": unit_meta["plant_name"],
        "tech": unit_meta["tech"],
        "zone": unit_meta["nyiso_zone"],
        "n_hours": int(len(p_all)),
        "n_hours_mar_nov": int(nw.sum()),
        "full_year_seas": clean(est_s),
        "full_year_annual_cap": clean(est_a),
        "full_year_oil_adj": clean(est_oil),
        "mar_nov_seas": clean(est_nw),
        "null_w0_blk24": clean(null0),
        "pos_blk24": pos,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summ_est_list(ests: list[dict], w_true: float | None = None) -> dict:
    if not ests:
        return {"n": 0}
    w = np.array([e["w_hat"] for e in ests], dtype=float)
    lo = np.array([e["w_lo"] for e in ests], dtype=float)
    hi = np.array([e["w_hi"] for e in ests], dtype=float)
    ident = np.array([e["ident"] for e in ests], dtype=bool)
    excl0 = np.array([e["excl0"] for e in ests], dtype=bool)
    both = np.array([e["two_sided"] for e in ests], dtype=bool)
    imp = np.array([e["sse_imp"] for e in ests], dtype=float)
    q = lambda x: [float(np.quantile(x, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    out = {
        "n": int(len(ests)),
        "frac_excl0": float(excl0.mean()),
        "frac_ident": float(ident.mean()),
        "frac_two_sided": float(both.mean()),
        "w_hat_q10_25_50_75_90": q(w),
        "w_lo_median": float(np.median(lo)),
        "w_hi_median": float(np.median(hi)),
        "w_hi_median_given_ident": float(np.median(hi[ident])) if ident.any() else None,
        "w_hat_median_given_two_sided": float(np.median(w[both])) if both.any() else None,
        "sse_imp_q10_50_90": [float(np.quantile(imp, p)) for p in (0.1, 0.5, 0.9)],
    }
    if w_true is not None:
        cov = (lo <= w_true + 1e-6) & (w_true - 1e-6 <= hi)
        out["coverage_5pct"] = float(cov.mean())
        if w_true > 0:
            relerr = np.abs(w - w_true) / w_true
            out["median_relerr_all"] = float(np.median(relerr))
            out["median_relerr_ident"] = float(np.median(relerr[ident])) if ident.any() else None
            out["median_relerr_two_sided"] = float(np.median(relerr[both])) if both.any() else None
    return out


def summarise_window_records(recs: list[dict], label: str) -> dict:
    """Summarise real, placebo, positive-control, and horse-race metrics for a window slice."""
    out = {
        "label": label,
        "n_windows": int(len(recs)),
        "real": _summ_est_list([r["real"] for r in recs]),
    }
    if not recs or "lbmp_shuffled" not in recs[0]:
        return out

    sh = [r["lbmp_shuffled"] for r in recs]
    out["lbmp_shuffled"] = _summ_est_list(sh)
    beats_min = np.array([r["real"]["sse_min"] < r["lbmp_shuffled"]["sse_min"] for r in recs], dtype=bool)
    beats_0 = np.array([r["real"]["sse0"] < r["lbmp_shuffled"]["sse0"] for r in recs], dtype=bool)
    k_win = int(beats_min.sum())
    pval = float(stats.binomtest(k_win, len(recs), 0.5, alternative="greater").pvalue) if len(recs) else 1.0
    out["lbmp_shuffled"]["paired_frac_real_beats_shuffled_sse_min"] = float(beats_min.mean())
    out["lbmp_shuffled"]["paired_frac_real_beats_shuffled_sse0"] = float(beats_0.mean())
    out["lbmp_shuffled"]["paired_binomial_pvalue"] = pval

    out["null_w0_blk24"] = _summ_est_list([r["null_blk24"] for r in recs], w_true=0.0)
    out["null_w0_iid"] = _summ_est_list([r["null_iid"] for r in recs], w_true=0.0)
    out["null_w0_gauss5pct"] = _summ_est_list([r["null_gauss5pct"] for r in recs], w_true=0.0)

    pos_summ = {}
    for wt in POS_W_LIST:
        key = str(int(wt))
        all_e = [r["pos_blk24"][key] for r in recs]
        cond_e = [r["pos_blk24"][key] for r in recs if r["max_margin"] > wt]
        pos_summ[key] = {
            "all_windows": _summ_est_list(all_e, w_true=wt),
            "cond_max_margin_gt_w": _summ_est_list(cond_e, w_true=wt),
        }
    out["positive_controls_blk24"] = pos_summ

    if "horse_race_real" in recs[0]:
        def summ_hr(hr_list, has_params=True):
            sk = np.array([h["sse_kink"] for h in hr_list], dtype=float)
            sq = np.array([h["sse_quad"] for h in hr_list], dtype=float)
            sc = np.array([h["sse_cap"] for h in hr_list], dtype=float)
            k_best = (sk <= sq) & (sk <= sc)
            q_best = (sq < sk) & (sq <= sc)
            c_best = (sc < sk) & (sc < sq)
            d = {
                "n": int(len(hr_list)),
                "frac_kink_beats_quad": float((sk < sq).mean()),
                "frac_kink_beats_cap": float((sk < sc).mean()),
                "frac_quad_beats_kink": float((sq < sk).mean()),
                "frac_cap_beats_kink": float((sc < sk).mean()),
                "best_of_3_kink": float(k_best.mean()),
                "best_of_3_quad": float(q_best.mean()),
                "best_of_3_cap": float(c_best.mean()),
                "median_rel_sse_diff_quad_minus_kink": float(np.median((sq - sk) / np.maximum(sq, 1e-12))),
                "median_rel_sse_diff_cap_minus_kink": float(np.median((sc - sk) / np.maximum(sc, 1e-12))),
            }
            if has_params:
                s0 = np.array([h["sse0"] for h in hr_list], dtype=float)
                d["median_sse_reduction_kink"] = float(np.median((s0 - sk) / np.maximum(s0, 1e-12)))
                d["median_sse_reduction_quad"] = float(np.median((s0 - sq) / np.maximum(s0, 1e-12)))
                d["median_sse_reduction_cap"] = float(np.median((s0 - sc) / np.maximum(s0, 1e-12)))
                d["median_w_hat"] = float(np.median([h["w_hat"] for h in hr_list]))
                d["median_gamma_hat"] = float(np.median([h["gamma_hat"] for h in hr_list]))
                d["median_theta_hat"] = float(np.median([h["theta_hat"] for h in hr_list]))
            return d

        out["horse_race_real"] = summ_hr([r["horse_race_real"] for r in recs], has_params=True)
        out["horse_race_pos30_control"] = summ_hr([r["horse_race_pos30"] for r in recs], has_params=False)
        ts_hr = [r["horse_race_real"] for r in recs if r["real"]["two_sided"]]
        if ts_hr:
            out["horse_race_real_given_two_sided"] = summ_hr(ts_hr, has_params=True)
    return out


def run_parallel_windows(units: list[dict], cfg: dict, workers: int = 6) -> list[dict]:
    tasks = [(u["meta"], u["arrays"], cfg) for u in units]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_worker_unit_windows, tasks))
    return [r for part in parts for r in part]


# ---------------------------------------------------------------------------
# Publication Figure (3 crisp panels, matched to R5 paper aesthetic)
# ---------------------------------------------------------------------------

def make_figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1e1c", "#52514e", "#e4e3df"
    BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    GRAY, LGRAY = "#6b6a66", "#a8a6a1"

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
    })

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(10.3, 2.85), gridspec_kw={"width_ratios": [1.08, 1.15, 0.98]}
    )

    # --- Panel (a): Empirical Unit Loading vs Net Profit Margin (Mar-Nov clean gas) ---
    mr = payload["margin_response"]
    ccgt_bins = mr["ccgt_mar_nov_seas"]
    st_bins = mr["st_mar_nov_seas"]

    def xy(blist):
        xs = np.array([b["m_mid"] for b in blist], dtype=float)
        med = np.array([b["median"] for b in blist], dtype=float)
        q25 = np.array([b["q25"] for b in blist], dtype=float)
        q75 = np.array([b["q75"] for b in blist], dtype=float)
        return xs, med, q25, q75

    xc, mc, q25c, q75c = xy(ccgt_bins)
    xs, ms, q25s, q75s = xy(st_bins)

    ax_a.fill_between(xc, q25c, q75c, color=BLUE, alpha=0.16, lw=0)
    ax_a.plot(xc, mc, "-o", color=BLUE, lw=1.6, ms=3.4, markeredgecolor="white",
              markeredgewidth=0.4, label="CCGT real (n=15, med & IQR)")
    ax_a.fill_between(xs, q25s, q75s, color=ORANGE, alpha=0.16, lw=0)
    ax_a.plot(xs, ms, "-s", color=ORANGE, lw=1.6, ms=3.4, markeredgecolor="white",
              markeredgewidth=0.4, label="Steam real (n=11, med & IQR)")

    # Overlay wear-kink LP prediction (w = $30/MWh, sigma = 0.85, pmin = 0.30)
    xk = np.array([-16.0, 0.0, 0.0, 30.0, 30.0, 110.0])
    yk = np.array([0.30, 0.30, 0.85, 0.85, 1.00, 1.00])
    ax_a.plot(xk, yk, "--", color=INK, lw=1.15, label=r"Wear-kink LP ($w=\$30$, $\sigma=0.85$)")
    ax_a.axhline(0.85, color=MUTED, ls=":", lw=0.8)
    ax_a.text(109.0, 0.832, r"$\sigma=0.85$", ha="right", va="top", fontsize=6.4, color=MUTED)

    ax_a.set_xlim(-18, 112)
    ax_a.set_xticks([0, 20, 40, 60, 80, 100])
    ax_a.set_ylim(0.20, 1.05)
    ax_a.set_xlabel(r"Net margin $\lambda_t - c_{g,t}$ (\$/MWh)")
    ax_a.set_ylabel(r"Unit loading $p_t / r_{g,\mathrm{seas}}$")
    ax_a.set_title("(a) Empirical dispatch vs. margin (Mar–Nov)", loc="left")
    ax_a.grid(axis="y", color=GRID, lw=0.6)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
                handlelength=1.5, borderpad=0.3)

    # --- Panel (b): Unit-Level Power vs Real Data & Placebos (168-h windows, Fig 4b analog) ---
    w168 = payload["windows_168h"]["all"]
    pos = w168["positive_controls_blk24"]
    items = [
        ("NYISO real (n=778)",
         w168["real"]["frac_two_sided"], w168["real"]["frac_excl0"], BLUE),
        ("LBMP shuffled",
         w168["lbmp_shuffled"]["frac_two_sided"], w168["lbmp_shuffled"]["frac_excl0"], GRAY),
        (r"Null $w=0$ (24-h blk)",
         w168["null_w0_blk24"]["frac_two_sided"], w168["null_w0_blk24"]["frac_excl0"], LGRAY),
        (r"Pos. $w=15$ ($m_{\max}>15$)",
         pos["15"]["cond_max_margin_gt_w"]["frac_two_sided"],
         pos["15"]["cond_max_margin_gt_w"]["frac_excl0"], GREEN),
        (r"Pos. $w=30$ ($m_{\max}>30$)",
         pos["30"]["cond_max_margin_gt_w"]["frac_two_sided"],
         pos["30"]["cond_max_margin_gt_w"]["frac_excl0"], GREEN),
        (r"Pos. $w=60$ ($m_{\max}>60$)",
         pos["60"]["cond_max_margin_gt_w"]["frac_two_sided"],
         pos["60"]["cond_max_margin_gt_w"]["frac_excl0"], GREEN),
        (r"Pos. $w=120$ ($m_{\max}>120$)",
         pos["120"]["cond_max_margin_gt_w"]["frac_two_sided"],
         pos["120"]["cond_max_margin_gt_w"]["frac_excl0"], GREEN),
    ]
    y = np.arange(len(items))
    h = 0.36
    vals_2s = [it[1] for it in items]
    vals_ex = [it[2] for it in items]
    cols = [it[3] for it in items]

    ax_b.barh(y - h / 2, vals_2s, height=h, color=cols, alpha=0.95)
    ax_b.barh(y + h / 2, vals_ex, height=h, color=cols, alpha=0.32, edgecolor=cols, lw=0.6)

    for yi, v2, ve in zip(y, vals_2s, vals_ex):
        ax_b.text(v2 + 0.015, yi - h / 2, f"{v2:.2f}", va="center", fontsize=6.3, color=INK, fontweight="bold")
        ax_b.text(ve + 0.015, yi + h / 2, f"{ve:.2f}", va="center", fontsize=6.2, color=MUTED)

    ax_b.set_yticks(y)
    ax_b.set_yticklabels([it[0] for it in items], fontsize=6.6)
    ax_b.tick_params(axis="y", length=0)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, 1.15)
    ax_b.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xlabel("Fraction of 168-h windows")
    ax_b.set_title("(b) Unit-level power vs. real & placebos", loc="left")
    ax_b.grid(axis="x", color=GRID, lw=0.6)
    ax_b.set_axisbelow(True)
    leg_b = [
        Patch(facecolor=MUTED, alpha=0.95, label=r"Two-sided ($0<w_{\rm lo}\leq w_{\rm hi}<300$)"),
        Patch(facecolor=MUTED, alpha=0.32, edgecolor=MUTED, lw=0.6, label=r"One-sided ($w_{\rm lo}>0$)"),
    ]
    ax_b.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.48, -0.25), ncol=2, frameon=False,
                fontsize=6.1, handlelength=1.1, columnspacing=0.9)

    # --- Panel (c): Equal-DOF (1 extra param) Model Horse Race (Real vs Planted w=30) ---
    hr_all = payload["windows_168h"]["all"]["horse_race_real"]
    hr_ccgt = payload["windows_168h"]["by_tech"]["CCGT"]["horse_race_real"]
    hr_st = payload["windows_168h"]["by_tech"]["ST"]["horse_race_real"]
    hr_pos = payload["windows_168h"]["all"]["horse_race_pos30_control"]

    cats = [
        "Real: All\n(n=778)",
        "Real: CCGT\n(n=491)",
        "Real: Steam\n(n=202)",
        "Planted\nw=30 ctrl",
    ]
    hrs = [hr_all, hr_ccgt, hr_st, hr_pos]
    k_share = np.array([h["best_of_3_kink"] for h in hrs]) * 100.0
    q_share = np.array([h["best_of_3_quad"] for h in hrs]) * 100.0
    c_share = np.array([h["best_of_3_cap"] for h in hrs]) * 100.0

    yc = np.arange(len(cats))
    bh = 0.56
    ax_c.barh(yc, k_share, height=bh, color=BLUE, label="Wear-kink LP ($w$)")
    ax_c.barh(yc, q_share, left=k_share, height=bh, color=ORANGE, label=r"Smooth quad. QP ($\gamma$)")
    ax_c.barh(yc, c_share, left=k_share + q_share, height=bh, color=GOLD, label=r"Const. ceiling ($\theta r$)")

    for yi, kv, qv, cv in zip(yc, k_share, q_share, c_share):
        if kv >= 11:
            ax_c.text(kv / 2, yi, f"{kv:.0f}%", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        if qv >= 11:
            ax_c.text(kv + qv / 2, yi, f"{qv:.0f}%", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        if cv >= 11:
            ax_c.text(kv + qv + cv / 2, yi, f"{cv:.0f}%", ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold")

    ax_c.axhline(2.5, color=MUTED, ls="--", lw=0.75)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(cats, fontsize=6.6)
    ax_c.tick_params(axis="y", length=0)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Best-of-3 SSE win rate (%, 1-DOF models)")
    ax_c.set_title("(c) Wear kink vs. non-wear alternatives", loc="left")
    ax_c.grid(axis="x", color=GRID, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.48, -0.25), ncol=3, frameon=False,
                fontsize=6.1, handlelength=1.0, columnspacing=0.8)

    fig.tight_layout(w_pad=1.3)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight", dpi=240)
    plt.close(fig)
    log(f"Saved figures: {OUT_FIG_PDF} and {OUT_FIG_PNG}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="R12 Unit-Level Real Market Data Discovery")
    parser.add_argument("--plot", action="store_true", help="Regenerate figure from cached JSON only")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    args = parser.parse_args()

    if args.plot:
        with open(OUT_JSON, encoding="utf-8") as fh:
            payload = json.load(fh)
        make_figure(payload)
        return 0

    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("== Step 0: Solver verification ==")
    solver_ver = verify_solvers()
    log("  Solver check:", solver_ver)

    log("== Step 1: Loading cached NY 2024 CEMS + EIA + NYISO datasets ==")
    primary_units, extended_units, funnel, utc_audit = build_unit_dataset()
    log(f"  Cohort funnel: {funnel['raw_ny_2024_cems_units']} raw -> {funnel['scale_ge100mw_and_ge2000h']} large -> "
        f"{funnel['exact_nyiso_node_and_ge10mo_eia923_fuel']} node+fuel -> "
        f"{funnel['dispatch_variability_std_ge008_primary_cohort']} primary non-cogen "
        f"(+{funnel['extended_cohort_including_cogen_ccgts'] - funnel['dispatch_variability_std_ge008_primary_cohort']} cogen CCGTs)")
    log(f"  UTC alignment: {utc_audit['n_merged_hours']} hrs, full-yr 0h corr="
        f"{utc_audit['all_ny_level_corr_vs_nyc_lbmp']['full_year']['+0h']:.4f}")

    log("== Step 2: Non-parametric margin-response & duct-firing diagnostics ==")
    margin_resp = compute_margin_response(primary_units)
    hr_kink_by_tech = {}
    for tech in ("CCGT", "ST", "GT"):
        sub = [u["meta"] for u in primary_units if u["meta"]["tech"] == tech and u["meta"]["delta_b_above_85"] is not None]
        if sub:
            dbs = np.array([m["delta_b_above_85"] for m in sub], dtype=float)
            hr_kink_by_tech[tech] = {
                "n_units": int(len(sub)),
                "median_b_below_85": float(np.median([m["b_below_85"] for m in sub])),
                "median_b_above_85": float(np.median([m["b_above_85"] for m in sub])),
                "median_delta_b": float(np.median(dbs)),
                "frac_delta_b_gt_1": float(np.mean(dbs > 1.0)),
            }
    log("  Duct-firing HR kink:", hr_kink_by_tech)

    log("== Step 3: Primary 168-h weekly windows (real + placebos + pos controls + 4-model horse race) ==")
    t1 = time.time()
    base_cfg_168 = {
        "win_h": 168, "sigma": 0.85, "cap_mode": "seasonal", "lbmp_mode": "nodal",
        "cost_mode": "gas", "cost_shift": 0.0, "do_placebos": True, "do_horse_race": True,
        "seed_base": 202400,
    }
    recs_168 = run_parallel_windows(primary_units, base_cfg_168, workers=args.workers)
    log(f"  Completed {len(recs_168)} 168-h windows in {time.time() - t1:.1f}s")

    win168_summary = {
        "all": summarise_window_records(recs_168, "All 168-h windows (28 units, full year)"),
        "mar_nov": summarise_window_records(
            [r for r in recs_168 if r["month"] not in (1, 2, 12)], "Mar-Nov clean gas (168-h)"
        ),
        "winter_djf": summarise_window_records(
            [r for r in recs_168 if r["month"] in (1, 2, 12)], "Winter DJF (168-h)"
        ),
        "by_tech": {
            t: summarise_window_records([r for r in recs_168 if r["tech"] == t], f"{t} (168-h)")
            for t in ("CCGT", "ST", "GT")
        },
        "by_season": {
            s: summarise_window_records([r for r in recs_168 if r["season"] == s], f"Season {s} (168-h)")
            for s in ("DJF", "MAM", "JJA", "SON")
        },
    }

    # Per-unit 168-h summary
    per_unit_168 = []
    for u in primary_units:
        uk = u["meta"]["unit_key"]
        ur = [r for r in recs_168 if r["unit"] == uk]
        if ur:
            sm = summarise_window_records(ur, uk)
            per_unit_168.append({
                "unit_key": uk,
                "plant_name": u["meta"]["plant_name"],
                "tech": u["meta"]["tech"],
                "zone": u["meta"]["nyiso_zone"],
                "n_windows_168h": len(ur),
                "real": sm["real"],
                "lbmp_shuffled_frac_two_sided": sm["lbmp_shuffled"]["frac_two_sided"],
                "real_beats_shuffled_sse_min": sm["lbmp_shuffled"]["paired_frac_real_beats_shuffled_sse_min"],
                "horse_race_best_of_3": {
                    "kink": sm["horse_race_real"]["best_of_3_kink"],
                    "quad": sm["horse_race_real"]["best_of_3_quad"],
                    "cap": sm["horse_race_real"]["best_of_3_cap"],
                },
            })

    log("  168-h Real summary:", {k: round(v, 3) if isinstance(v, float) else v for k, v in win168_summary["all"]["real"].items() if k in ("n", "frac_excl0", "frac_ident", "frac_two_sided")})
    log("  168-h Shuf summary:", {k: round(v, 3) if isinstance(v, float) else v for k, v in win168_summary["all"]["lbmp_shuffled"].items() if k in ("frac_excl0", "frac_ident", "frac_two_sided", "paired_frac_real_beats_shuffled_sse_min")})
    log("  168-h Horse race REAL :", {k: round(v, 3) for k, v in win168_summary["all"]["horse_race_real"].items() if "best_of_3" in k or "median_sse_reduction" in k})
    log("  168-h Horse race POS30:", {k: round(v, 3) for k, v in win168_summary["all"]["horse_race_pos30_control"].items() if "best_of_3" in k})

    log("== Step 4: 72-h windows (parallel to R5 window length) ==")
    t2 = time.time()
    cfg_72 = {
        "win_h": 72, "sigma": 0.85, "cap_mode": "seasonal", "lbmp_mode": "nodal",
        "cost_mode": "gas", "cost_shift": 0.0, "do_placebos": True, "do_horse_race": False,
        "seed_base": 720000,
    }
    recs_72 = run_parallel_windows(primary_units, cfg_72, workers=args.workers)
    win72_summary = {
        "all": summarise_window_records(recs_72, "All 72-h windows (28 units, full year)"),
        "mar_nov": summarise_window_records([r for r in recs_72 if r["month"] not in (1, 2, 12)], "Mar-Nov (72-h)"),
        "by_tech": {
            t: summarise_window_records([r for r in recs_72 if r["tech"] == t], f"{t} (72-h)")
            for t in ("CCGT", "ST", "GT")
        },
    }
    log(f"  Completed {len(recs_72)} 72-h windows in {time.time() - t2:.1f}s")

    log("== Step 5: Full-year unit-level inverse optimization (28 units) ==")
    t3 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fy_units = list(ex.map(_worker_unit_fullyear, [(u["meta"], u["arrays"]) for u in primary_units]))
    fy_summary = {
        "full_year_seas": _summ_est_list([u["full_year_seas"] for u in fy_units]),
        "full_year_annual_cap": _summ_est_list([u["full_year_annual_cap"] for u in fy_units]),
        "full_year_oil_adj": _summ_est_list([u["full_year_oil_adj"] for u in fy_units]),
        "mar_nov_seas": _summ_est_list([u["mar_nov_seas"] for u in fy_units]),
        "null_w0_blk24": _summ_est_list([u["null_w0_blk24"] for u in fy_units], w_true=0.0),
        "pos_blk24": {
            str(int(wt)): _summ_est_list([u["pos_blk24"][str(int(wt))] for u in fy_units], w_true=wt)
            for wt in (15.0, 30.0, 60.0)
        },
        "per_unit": fy_units,
    }
    log(f"  Completed full-year profiles for {len(fy_units)} units in {time.time() - t3:.1f}s")

    log("== Step 6: Sensitivity suite (168-h windows) ==")
    t4 = time.time()
    sens_specs = [
        ("sigma=0.80",         primary_units, {"sigma": 0.80}),
        ("sigma=0.90",         primary_units, {"sigma": 0.90}),
        ("cost_shift=-10",     primary_units, {"cost_shift": -10.0}),
        ("cost_shift=+10",     primary_units, {"cost_shift": +10.0}),
        ("cap=annual_p995",    primary_units, {"cap_mode": "annual"}),
        ("lbmp=zonal",         primary_units, {"lbmp_mode": "zonal"}),
        ("cost=oil_adjusted",  primary_units, {"cost_mode": "oil_adj"}),
        (f"cohort={len(extended_units)}_incl_cogen", extended_units, {}),
    ]
    sens_out = {
        "baseline_sigma085_seas_nodal": win168_summary["all"]["real"],
    }
    for name, ulist, overrides in sens_specs:
        cfg_s = {
            "win_h": 168, "sigma": 0.85, "cap_mode": "seasonal", "lbmp_mode": "nodal",
            "cost_mode": "gas", "cost_shift": 0.0, "do_placebos": False, "do_horse_race": False,
            "seed_base": 990000,
        }
        cfg_s.update(overrides)
        rs = run_parallel_windows(ulist, cfg_s, workers=args.workers)
        sm = _summ_est_list([r["real"] for r in rs])
        # Also report CCGT-only for ambient derating clarity
        sm_ccgt = _summ_est_list([r["real"] for r in rs if r["tech"] == "CCGT"])
        sm["ccgt_subset"] = {
            "n": sm_ccgt["n"],
            "frac_excl0": sm_ccgt["frac_excl0"],
            "frac_ident": sm_ccgt["frac_ident"],
            "frac_two_sided": sm_ccgt["frac_two_sided"],
            "w_hat_median": sm_ccgt["w_hat_q10_25_50_75_90"][2],
        }
        sens_out[name] = sm
        log(f"  {name:24s}: n={sm['n']:3d}, excl0={sm['frac_excl0']:.3f}, ident={sm['frac_ident']:.3f}, "
            f"2s={sm['frac_two_sided']:.3f}, med_w={sm['w_hat_q10_25_50_75_90'][2]:.1f} "
            f"(CCGT med_w={sm['ccgt_subset']['w_hat_median']:.1f})")
    log(f"  Completed sensitivity suite in {time.time() - t4:.1f}s")

    payload = {
        "description": "R12 Stage 3 Unit-Level Real Market Data Discovery (NYISO 2024 + EPA CEMS + EIA-860/923)",
        "runtime_seconds": float(time.time() - t_start),
        "solver_verification": solver_ver,
        "utc_alignment_verification": utc_audit,
        "cohort_funnel": funnel,
        "units_metadata": [u["meta"] for u in primary_units],
        "duct_firing_hr_kink_by_tech": hr_kink_by_tech,
        "margin_response": margin_resp,
        "windows_168h": win168_summary,
        "per_unit_168h": per_unit_168,
        "windows_72h": win72_summary,
        "full_year": fy_summary,
        "sensitivity_168h": sens_out,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    log(f"Wrote {OUT_JSON} (total runtime {time.time() - t_start:.1f}s)")

    make_figure(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
