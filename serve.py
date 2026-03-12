import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import tensorrt as trt

from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List

import asyncio

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from build_trt_engine import load_engine, create_context, run_inference

# ---------------------------------------------------------------------------
# Constants (must match training config)
# ---------------------------------------------------------------------------
ENGINE_PATH = "models/tft_model.trt"
DATASET_PARAMS_PATH = "models/dataset_params.pt"

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 7

KP_FEATURES = [
    "Kp_0_3", "Kp_3_6", "Kp_6_9", "Kp_9_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
]
AP_FEATURES = [
    "Ap_0_3", "Ap_3_6", "Ap_6_9", "Ap_9_12",
    "Ap_12_15", "Ap_15_18", "Ap_18_21", "Ap_21_24",
]
CYCLE_STARTS = pd.to_datetime([
    "1933-09-01", "1944-02-01", "1954-04-01", "1964-10-01",
    "1976-06-01", "1986-09-01", "1996-05-01", "2008-12-01", "2019-12-01",
])

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app_state: dict = {}
gpu_semaphore = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Lifespan: load engine once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = load_engine(ENGINE_PATH)
    app_state["engine"] = engine
    app_state["context"] = create_context(engine)  # created once, reused per request
    app_state["dataset_params"] = torch.load(DATASET_PARAMS_PATH, weights_only=False)
    print("TensorRT engine, context, and dataset params loaded.")
    yield
    app_state.clear()


app = FastAPI(title="TFT Solar Flux Inference", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class DailyRecord(BaseModel):
    yy: int
    mm: int
    dd: int
    ND: float
    Cp: float
    C9: float
    Obs_F10_7: float
    Kp_0_3: float; Kp_3_6: float; Kp_6_9: float; Kp_9_12: float
    Kp_12_15: float; Kp_15_18: float; Kp_18_21: float; Kp_21_24: float
    Ap_0_3: float; Ap_3_6: float; Ap_6_9: float; Ap_9_12: float
    Ap_12_15: float; Ap_15_18: float; Ap_18_21: float; Ap_21_24: float


class PredictRequest(BaseModel):
    # Must contain at least MAX_ENCODER_LENGTH rows of history
    history: List[DailyRecord]


class PredictResponse(BaseModel):
    # 7 forecast days x 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    quantiles: List[List[float]]
    forecast_dates: List[str]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"].values)
    no_cycle = ts < CYCLE_STARTS[0]
    cycle_idx = np.searchsorted(CYCLE_STARTS, ts, side="right") - 1
    cycle_idx = np.clip(cycle_idx, 0, len(CYCLE_STARTS) - 1)

    elapsed_days = np.array((ts - CYCLE_STARTS[cycle_idx]).days, dtype=float)
    years_elapsed = elapsed_days / 365.25
    years_elapsed[no_cycle] = 0.0

    df["years_since_min"] = years_elapsed
    df["cycle_sin"] = np.sin(2 * np.pi * years_elapsed / 11.0)
    df["cycle_cos"] = np.cos(2 * np.pi * years_elapsed / 11.0)
    raw_cycle_num = 17 + cycle_idx
    df["cycle_num"] = np.clip(raw_cycle_num, 16, 24).astype(str)
    df.loc[no_cycle, "cycle_num"] = "16"
    return df


def build_dataframe(records: List[DailyRecord]) -> pd.DataFrame:
    data = defaultdict(list)
    for r in records:
        for field, val in r.model_dump().items():
            data[field].append(val)

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(
        df[["yy", "mm", "dd"]].rename(columns={"yy": "year", "mm": "month", "dd": "day"})
    )
    df["time_idx"] = np.arange(len(df))
    df["group_ids"] = 1
    df["flux_lag_27"] = df["Obs_F10_7"].shift(27).bfill()
    df = add_cycle_features(df)
    return df


def build_dataloader(df: pd.DataFrame, dataset_params: dict):
    dataset = TimeSeriesDataSet.from_parameters(dataset_params, df, predict=True)
    return dataset.to_dataloader(train=False, batch_size=1, num_workers=0)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if len(request.history) < MAX_ENCODER_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"At least {MAX_ENCODER_LENGTH} days of history required, got {len(request.history)}.",
        )

    try:
        df = build_dataframe(request.history)
        dataloader = build_dataloader(df, app_state["dataset_params"])
        batch, _ = next(iter(dataloader))

        # Convert batch dict to ordered tensors matching TensorRT input names
        context = app_state["context"]
        engine  = context.engine
        input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        inputs = {name: batch[name] for name in input_names}

        async with gpu_semaphore:
            output = await run_inference(context, inputs)  # shape: (1, 7, 7)
        quantiles = output[0].tolist()          # (7 forecast steps, 7 quantiles)

        last_date = pd.Timestamp(
            year=request.history[-1].yy,
            month=request.history[-1].mm,
            day=request.history[-1].dd,
        )
        forecast_dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(MAX_PREDICTION_LENGTH)
        ]

        return PredictResponse(quantiles=quantiles, forecast_dates=forecast_dates)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
@app.get("/ping")           # SageMaker health check contract
async def health():
    return {"status": "ok", "engine_loaded": "engine" in app_state}


@app.post("/invocations")   # SageMaker inference contract
async def invocations(request: PredictRequest):
    return await predict(request)