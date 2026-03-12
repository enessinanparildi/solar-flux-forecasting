"""
Ray Serve deployment for TFT solar flux inference.

Each replica owns one GPU exclusively — no semaphore needed.
Scale replicas to match GPU count.

Usage (local):
    python serve_ray.py

Usage (multi-GPU, 3 replicas):
    NUM_REPLICAS=3 python serve_ray.py

SageMaker:
    Exposes /ping and /invocations to satisfy the SageMaker inference contract.
"""

import os
import logging
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import tensorrt as trt

import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from build_trt_engine import load_engine, create_context, _run_inference_sync
from typing import List as BatchList  # alias to avoid shadowing inside @serve.batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match training config)
# ---------------------------------------------------------------------------
ENGINE_PATH         = os.environ.get("ENGINE_PATH",         "models/tft_model.trt")
DATASET_PARAMS_PATH = os.environ.get("DATASET_PARAMS_PATH", "models/dataset_params.pt")
NUM_REPLICAS        = int(os.environ.get("NUM_REPLICAS",    "1"))
MAX_BATCH_SIZE      = int(os.environ.get("MAX_BATCH_SIZE",  "64"))
BATCH_WAIT_MS       = float(os.environ.get("BATCH_WAIT_MS", "20"))  # ms to wait before flushing

MAX_ENCODER_LENGTH    = 60
MAX_PREDICTION_LENGTH = 7

CYCLE_STARTS = pd.to_datetime([
    "1933-09-01", "1944-02-01", "1954-04-01", "1964-10-01",
    "1976-06-01", "1986-09-01", "1996-05-01", "2008-12-01", "2019-12-01",
])

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
    Kp_0_3: float;  Kp_3_6: float;  Kp_6_9: float;  Kp_9_12: float
    Kp_12_15: float; Kp_15_18: float; Kp_18_21: float; Kp_21_24: float
    Ap_0_3: float;  Ap_3_6: float;  Ap_6_9: float;  Ap_9_12: float
    Ap_12_15: float; Ap_15_18: float; Ap_18_21: float; Ap_21_24: float


class PredictRequest(BaseModel):
    history: List[DailyRecord]


class PredictResponse(BaseModel):
    quantiles: List[List[float]]
    forecast_dates: List[str]


# ---------------------------------------------------------------------------
# Preprocessing helpers (identical to serve.py)
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


def build_dataloader(df: pd.DataFrame, dataset_params: dict, batch_size: int = 1):
    dataset = TimeSeriesDataSet.from_parameters(dataset_params, df, predict=True)
    return dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


def preprocess_request(request: "PredictRequest", dataset_params: dict) -> dict:
    """Preprocess a single request into a tensor dict (batch_size=1)."""
    df = build_dataframe(request.history)
    dataloader = build_dataloader(df, dataset_params, batch_size=1)
    batch, _ = next(iter(dataloader))
    return batch


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------
app = FastAPI(title="TFT Solar Flux Inference (Ray Serve)")


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=MAX_BATCH_SIZE,
)
@serve.ingress(app)
class TFTDeployment:
    def __init__(self):
        self.engine         = load_engine(ENGINE_PATH)
        self.context        = create_context(self.engine)
        self.dataset_params = torch.load(DATASET_PARAMS_PATH, weights_only=False)
        self.input_names    = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        log.info("TensorRT engine loaded in replica.")

    @serve.batch(max_batch_size=MAX_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_MS / 1000)
    async def _predict_batch(
        self, requests: BatchList[PredictRequest]
    ) -> BatchList[PredictResponse]:
        # Preprocess each request independently (CPU-side, parallelisable)
        batches = [preprocess_request(r, self.dataset_params) for r in requests]

        # Stack tensors along batch dim → one TRT call for the whole batch
        inputs = {
            name: torch.cat([b[name] for b in batches], dim=0)
            for name in self.input_names
        }

        output = _run_inference_sync(self.context, inputs)  # (N, 7, 7)

        responses = []
        for i, req in enumerate(requests):
            last = req.history[-1]
            last_date = pd.Timestamp(year=last.yy, month=last.mm, day=last.dd)
            forecast_dates = [
                (last_date + pd.Timedelta(days=j + 1)).strftime("%Y-%m-%d")
                for j in range(MAX_PREDICTION_LENGTH)
            ]
            responses.append(PredictResponse(quantiles=output[i].tolist(), forecast_dates=forecast_dates))

        return responses

    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        # Validate before entering the batch queue so bad requests fail fast
        if len(request.history) < MAX_ENCODER_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"At least {MAX_ENCODER_LENGTH} days of history required, got {len(request.history)}.",
            )
        try:
            return await self._predict_batch(request)  # type: ignore[arg-type, return-value]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    @app.get("/ping")           # SageMaker health check contract
    def health(self):
        return {"status": "ok"}

    @app.post("/invocations")   # SageMaker inference contract
    async def invocations(self, request: PredictRequest) -> PredictResponse:
        return await self.predict(request)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))  # SageMaker requires 8080
    ray.init()
    serve.run(TFTDeployment.bind(), host="0.0.0.0", port=port)