"""
Feast feature definitions for solar flux forecasting.

Features are computed once during ingestion (ingest.py) and stored in the
offline store. Both training and serving read from the same definitions,
eliminating training-serving skew.

Apply changes with:
    cd feature_repo && feast apply
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, String

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
# Solar flux is a single global time series — one entity covers all rows.
data_source = Entity(
    name="data_source",
    description="Solar data provider identifier (always 'NOAA' for this project).",
)

# ---------------------------------------------------------------------------
# Offline data source — local Parquet file populated by ingest.py
# ---------------------------------------------------------------------------
solar_file_source = FileSource(
    path="../data/feast/solar_flux.parquet",
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature view
# ---------------------------------------------------------------------------
solar_daily_fv = FeatureView(
    name="solar_daily_features",
    entities=[data_source],
    ttl=timedelta(days=365 * 30),   # 30 years of history retained
    schema=[
        # --- Raw measurements ---
        Field(name="obs_f10_7",   dtype=Float32),
        Field(name="nd",          dtype=Float32),
        Field(name="cp",          dtype=Float32),
        Field(name="c9",          dtype=Float32),
        # Kp 3-hour intervals
        Field(name="kp_0_3",      dtype=Float32),
        Field(name="kp_3_6",      dtype=Float32),
        Field(name="kp_6_9",      dtype=Float32),
        Field(name="kp_9_12",     dtype=Float32),
        Field(name="kp_12_15",    dtype=Float32),
        Field(name="kp_15_18",    dtype=Float32),
        Field(name="kp_18_21",    dtype=Float32),
        Field(name="kp_21_24",    dtype=Float32),
        # Ap 3-hour intervals
        Field(name="ap_0_3",      dtype=Float32),
        Field(name="ap_3_6",      dtype=Float32),
        Field(name="ap_6_9",      dtype=Float32),
        Field(name="ap_9_12",     dtype=Float32),
        Field(name="ap_12_15",    dtype=Float32),
        Field(name="ap_15_18",    dtype=Float32),
        Field(name="ap_18_21",    dtype=Float32),
        Field(name="ap_21_24",    dtype=Float32),
        # --- Engineered features (computed at ingestion time) ---
        Field(name="flux_lag_27",     dtype=Float32),
        Field(name="years_since_min", dtype=Float32),
        Field(name="cycle_sin",       dtype=Float32),
        Field(name="cycle_cos",       dtype=Float32),
        Field(name="cycle_num",       dtype=String),
        # Calendar fields used as time-varying knowns in TFT
        Field(name="yy",              dtype=Float32),
        Field(name="mm",              dtype=Float32),
        Field(name="dd",              dtype=Float32),
    ],
    source=solar_file_source,
)

# ---------------------------------------------------------------------------
# Feature service — logical grouping consumed by training and serving
# ---------------------------------------------------------------------------
solar_feature_service = FeatureService(
    name="solar_training_features",
    features=[solar_daily_fv],
)