"""
Ingest solar flux data into the Feast offline (and optionally online) store.

Run once to bootstrap, then nightly via cron/Airflow to pick up new NOAA data.

Usage:
    # From the project root:
    python feature_repo/ingest.py                    # ingest from local file
    python feature_repo/ingest.py --materialize      # also push latest to online store
    python feature_repo/ingest.py --source noaa      # fetch latest from NOAA API
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from feast import FeatureStore

RAW_DATA_PATH   = os.environ.get("RAW_DATA_PATH",   "data/SW-All.txt")
PARQUET_PATH    = os.environ.get("PARQUET_PATH",    "data/feast/solar_flux.parquet")
FEATURE_REPO    = os.environ.get("FEATURE_REPO",    "feature_repo")
NOAA_URL        = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/solar-radio/sg-discrete/solar-radio-flux-observed-smoothed/sg-discrete.txt"

CYCLE_STARTS = pd.to_datetime([
    "1933-09-01", "1944-02-01", "1954-04-01", "1964-10-01",
    "1976-06-01", "1986-09-01", "1996-05-01", "2008-12-01", "2019-12-01",
])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sw_file(path: str) -> pd.DataFrame:
    """Parse the NOAA SW-All.txt fixed-width format into a DataFrame."""
    with open(path, "r") as f:
        lines = f.readlines()

    begin_num = end_num = 0
    for num, line in enumerate(lines):
        if line == "BEGIN OBSERVED\n":
            begin_num = num
        if line == "END OBSERVED\n":
            end_num = num

    line_data = lines[begin_num + 1: end_num]

    cols = [
        "yy", "mm", "dd", "BSRN", "ND",
        "Kp", "Kp", "Kp", "Kp", "Kp", "Kp", "Kp", "Kp",
        "Sum", "Ap", "Ap", "Ap", "Ap", "Ap", "Ap", "Ap", "Ap",
        "Avg", "Cp", "C9", "ISN", "Adj_F10_7", "Q", "Adj_Ctr81", "Adj_Lst81",
        "Obs_F10_7", "Obs_Ctr81", "Obs_Lst81",
    ]
    suffixes = [f"{i}_{i+3}" for i in range(0, 24, 3)]
    out_cols, kp_idx, ap_idx = [], 0, 0
    for c in cols:
        if c == "Kp":
            out_cols.append(f"Kp_{suffixes[kp_idx]}"); kp_idx += 1
        elif c == "Ap":
            out_cols.append(f"Ap_{suffixes[ap_idx]}"); ap_idx += 1
        else:
            out_cols.append(c)

    data_dict = defaultdict(list)
    for line in line_data:
        row = [v for v in line[:-1].split(" ") if v]
        for i, val in enumerate(row):
            data_dict[out_cols[i]].append(val)

    df = pd.DataFrame(data_dict)
    for col in out_cols:
        df[col] = df[col].astype(float)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (single source of truth — used by training and serving)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df[["yy", "mm", "dd"]].rename(columns={"yy": "year", "mm": "month", "dd": "day"})
    )

    # Lag feature
    df["flux_lag_27"] = df["Obs_F10_7"].shift(27).bfill()

    # Solar cycle features
    ts = df["timestamp"].values
    ts_pd = pd.to_datetime(ts)
    no_cycle = ts_pd < CYCLE_STARTS[0]
    cycle_idx = np.searchsorted(CYCLE_STARTS, ts_pd, side="right") - 1
    cycle_idx = np.clip(cycle_idx, 0, len(CYCLE_STARTS) - 1)

    elapsed_days = np.array((ts_pd - CYCLE_STARTS[cycle_idx]).days, dtype=float)
    years_elapsed = elapsed_days / 365.25
    years_elapsed[no_cycle] = 0.0

    df["years_since_min"] = years_elapsed
    df["cycle_sin"] = np.sin(2 * np.pi * years_elapsed / 11.0)
    df["cycle_cos"] = np.cos(2 * np.pi * years_elapsed / 11.0)
    df["cycle_num"] = np.clip(17 + cycle_idx, 16, 24).astype(str)
    df.loc[no_cycle, "cycle_num"] = "16"

    return df


def to_feast_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to lowercase Feast conventions and add required fields."""
    rename = {
        "Obs_F10_7": "obs_f10_7", "ND": "nd", "Cp": "cp", "C9": "c9",
        "Kp_0_3":  "kp_0_3",   "Kp_3_6":  "kp_3_6",   "Kp_6_9":  "kp_6_9",   "Kp_9_12":  "kp_9_12",
        "Kp_12_15": "kp_12_15", "Kp_15_18": "kp_15_18", "Kp_18_21": "kp_18_21", "Kp_21_24": "kp_21_24",
        "Ap_0_3":  "ap_0_3",   "Ap_3_6":  "ap_3_6",   "Ap_6_9":  "ap_6_9",   "Ap_9_12":  "ap_9_12",
        "Ap_12_15": "ap_12_15", "Ap_15_18": "ap_15_18", "Ap_18_21": "ap_18_21", "Ap_21_24": "ap_21_24",
    }
    df = df.rename(columns=rename)

    # Feast requires event_timestamp (UTC) and an entity column
    df["event_timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df["data_source"] = "NOAA"

    feature_cols = list(rename.values()) + [
        "flux_lag_27", "years_since_min", "cycle_sin", "cycle_cos", "cycle_num",
        "yy", "mm", "dd",
    ]
    return df[["event_timestamp", "data_source"] + feature_cols]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_from_file(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    print(f"Parsing {path}...")
    df = parse_sw_file(path)
    df = compute_features(df)
    return to_feast_schema(df)


def ingest_from_noaa() -> pd.DataFrame:
    """Fetch the latest daily values directly from NOAA and return a feast-ready DataFrame."""
    print(f"Fetching from NOAA: {NOAA_URL}")
    import urllib.request
    with urllib.request.urlopen(NOAA_URL) as resp:
        content = resp.read().decode()

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        return ingest_from_file(tmp)
    finally:
        os.unlink(tmp)


def write_parquet(df: pd.DataFrame, path: str = PARQUET_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")


def materialize(store: FeatureStore) -> None:
    """Push latest feature values from offline → online store."""
    end   = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=90)   # 90-day window covers the 60-day encoder + buffer
    print(f"Materializing {start.date()} → {end.date()} into online store...")
    store.materialize(start_date=start, end_date=end)
    print("Materialization complete.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      choices=["file", "noaa"], default="file")
    parser.add_argument("--materialize", action="store_true", help="Push latest to online store after ingestion")
    args = parser.parse_args()

    df = ingest_from_noaa() if args.source == "noaa" else ingest_from_file()
    write_parquet(df)

    store = FeatureStore(repo_path=FEATURE_REPO)

    if args.materialize:
        materialize(store)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()