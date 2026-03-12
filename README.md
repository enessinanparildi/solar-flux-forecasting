# Solar Flux Forecasting with Temporal Fusion Transformer

A production-grade 7-day solar radio flux (F10.7) forecasting system using a **Temporal Fusion Transformer**, served via **TensorRT** + **FastAPI**, and deployable to **AWS SageMaker**.

---

## Overview

Solar radio flux (F10.7 index) is a key indicator of solar activity used in space weather forecasting, satellite drag modeling, and ionospheric research. This project trains a probabilistic TFT model on 90+ years of NOAA daily solar observations and serves real-time 7-day quantile forecasts with sub-millisecond GPU inference.

---

## Architecture

```
NOAA Data (SW-All.txt)
        │
        ▼
  feature_repo/ingest.py        ← feature engineering + Feast offline store
        │
        ▼
  tft_model.py                  ← TFT training (PyTorch Forecasting + Lightning)
        │
        ▼
  export_to_onnx.py             ← export to ONNX
        │
        ▼
  build_trt_engine.py           ← compile TensorRT FP16 engine
        │
        ▼
  serve.py / serve_ray.py       ← FastAPI inference server
        │
        ▼
  Docker + SageMaker            ← containerized deployment
```

---

## Model

| Property | Value |
|---|---|
| Architecture | Temporal Fusion Transformer |
| Target | F10.7 solar radio flux (observed) |
| Forecast horizon | 7 days |
| Encoder history | 60 days |
| Output | 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] |
| Loss | Quantile loss |
| Optimizer | Ranger |
| Precision | FP16 mixed |
| Hidden size | 256 |
| Attention heads | 16 |

### Input Features

**Time-varying unknowns** (observed up to forecast date):
- `Obs_F10_7` — observed solar flux
- `flux_lag_27` — 27-day lag (solar rotation period)
- Kp index — 8 × 3-hour geomagnetic activity intervals
- Ap index — 8 × 3-hour geomagnetic activity intervals
- `ND`, `Cp`, `C9` — geomagnetic disturbance indicators

**Time-varying knowns** (known into the future):
- `yy`, `mm`, `dd` — calendar date
- `cycle_sin`, `cycle_cos` — solar cycle phase (11-year period, sinusoidal encoding)

**Static categorical:**
- `cycle_num` — solar cycle number (16–24)

---

## Feature Store (Feast)

Features are computed once at ingestion and stored in a [Feast](https://feast.dev/) offline store, providing a single source of truth for both training and serving.

```bash
# Populate offline store from local file
python feature_repo/ingest.py

# Fetch latest data directly from NOAA and ingest
python feature_repo/ingest.py --source noaa

# Also materialize to online store
python feature_repo/ingest.py --materialize

# Register feature definitions
cd feature_repo && feast apply
```

---

## Inference Server

The model is compiled to a TensorRT FP16 engine for optimized GPU inference and served via FastAPI.

**Endpoint:** `POST /predict`

```json
{
  "history": [
    {
      "yy": 2024, "mm": 4, "dd": 30,
      "Obs_F10_7": 152.3,
      "ND": 1.0, "Cp": 0.3, "C9": 2.0,
      "Kp_0_3": 2.0, "Kp_3_6": 1.7, "...",
      "Ap_0_3": 7.0, "Ap_3_6": 5.0, "..."
    }
  ]
}
```

Requires at least 60 days of history. Returns:

```json
{
  "forecast_dates": ["2024-05-01", "2024-05-02", "..."],
  "quantiles": [
    [q02, q10, q25, q50, q75, q90, q98],
    "..."
  ]
}
```

---

## Deployment

### Local

```bash
pip install -r requirements.txt

# Build TRT engine (first time only)
python build_trt_engine.py

# Start server
uvicorn serve:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t solar-flux-tft .
docker run --gpus all -p 8080:8080 solar-flux-tft
```

The container auto-builds the TRT engine from the bundled ONNX model at startup if no pre-built engine is found (required since TRT engines are GPU-architecture specific).

### AWS SageMaker

The server implements the SageMaker inference contract:

- `GET /ping` — health check
- `POST /invocations` — inference endpoint

Deploy by pushing the Docker image to ECR and creating a SageMaker endpoint with a GPU instance (e.g., `ml.g4dn.xlarge`).

---

## Project Structure

```
├── tft_model.py              # Training, dataset construction, evaluation
├── export_to_onnx.py         # PyTorch → ONNX export
├── build_trt_engine.py       # ONNX → TensorRT engine + inference runtime
├── serve.py                  # FastAPI inference server
├── serve_ray.py              # Ray Serve variant
├── feature_repo/
│   ├── feature_store.yaml    # Feast configuration
│   ├── features.py           # Entity, FeatureView, FeatureService definitions
│   └── ingest.py             # Data ingestion + feature engineering pipeline
├── models/                   # ONNX + TRT engine artifacts
├── data/                     # Raw NOAA data + Feast offline store
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

- **Model:** [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) · [PyTorch Lightning](https://lightning.ai/)
- **Inference:** [TensorRT](https://developer.nvidia.com/tensorrt) FP16
- **Serving:** [FastAPI](https://fastapi.tiangolo.com/) · [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- **Feature Store:** [Feast](https://feast.dev/)
- **Experiment Tracking:** [MLflow](https://mlflow.org/)
- **Deployment:** Docker · AWS SageMaker

---

## Data

Solar geophysical data sourced from [NOAA National Centers for Environmental Information](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/). Daily observations from 1932 to present covering solar radio flux, geomagnetic indices (Kp, Ap), and disturbance indicators.