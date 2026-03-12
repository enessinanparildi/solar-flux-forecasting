import os

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Heavy GPU/TRT/Ray deps — only available on machines with CUDA + TensorRT
# ---------------------------------------------------------------------------
try:
    import torch
    import ray
    from ray import serve
    from serve_ray import TFTDeployment
    from tft_model import read_csv
    HAS_GPU_DEPS = torch.cuda.is_available()
except ImportError:
    HAS_GPU_DEPS = False

requires_gpu = pytest.mark.skipif(not HAS_GPU_DEPS, reason="GPU/TRT/Ray deps not available")

_BASE = os.environ.get("PERCEPTIVESPACE_DIR", "D:/perceptivespace")

DAILY_RECORD_FIELDS = [
    "yy", "mm", "dd", "ND", "Cp", "C9", "Obs_F10_7",
    "Kp_0_3", "Kp_3_6", "Kp_6_9", "Kp_9_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "Ap_0_3", "Ap_3_6", "Ap_6_9", "Ap_9_12",
    "Ap_12_15", "Ap_15_18", "Ap_18_21", "Ap_21_24",
]


def df_to_records(df, n_rows=67):
    rows = df.tail(n_rows)
    return [{field: float(row[field]) for field in DAILY_RECORD_FIELDS} for _, row in rows.iterrows()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ray_serve():
    """Start a Ray Serve instance for the session, shut it down after."""
    if not HAS_GPU_DEPS:
        pytest.skip("GPU/TRT/Ray deps not available")

    ray.init(ignore_reinit_error=True)
    serve.run(TFTDeployment.bind(), host="127.0.0.1", port=8000)

    yield

    serve.shutdown()
    ray.shutdown()


@pytest.fixture(scope="session")
def records():
    _, _, test_df, _ = read_csv()
    return df_to_records(test_df, n_rows=67)


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

@requires_gpu
def test_health(ray_serve):
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@requires_gpu
def test_predict(ray_serve, records):
    resp = requests.post(f"{BASE_URL}/predict", json={"history": records})
    assert resp.status_code == 200, f"Server error: {resp.text}"

    data = resp.json()
    assert len(data["forecast_dates"]) == 7
    assert len(data["quantiles"]) == 7
    assert all(len(q) == 7 for q in data["quantiles"])


@requires_gpu
def test_predict_quantiles_ordered(ray_serve, records):
    """Median should be between the outer quantiles for every forecast step."""
    resp = requests.post(f"{BASE_URL}/predict", json={"history": records})
    assert resp.status_code == 200

    # quantiles order: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    for step in resp.json()["quantiles"]:
        assert step[0] <= step[3] <= step[6], (
            f"Quantile ordering violated: {step}"
        )


@requires_gpu
def test_too_few_records(ray_serve):
    resp = requests.post(f"{BASE_URL}/predict", json={"history": [
        {field: 0.0 for field in DAILY_RECORD_FIELDS} for _ in range(5)
    ]})
    assert resp.status_code == 422


@requires_gpu
def test_invocations(ray_serve, records):
    """SageMaker inference contract endpoint."""
    resp = requests.post(f"{BASE_URL}/invocations", json={"history": records})
    assert resp.status_code == 200
    data = resp.json()
    assert "forecast_dates" in data
    assert "quantiles" in data


@requires_gpu
def test_ping(ray_serve):
    """SageMaker health check contract endpoint."""
    resp = requests.get(f"{BASE_URL}/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@requires_gpu
def test_batching(ray_serve, records):
    """Fire concurrent requests and verify all responses are valid.

    Ray Serve batches them into a single TRT forward pass when they
    arrive within BATCH_WAIT_MS of each other.
    """
    import concurrent.futures

    N = 8  # concurrent requests

    def call():
        return requests.post(f"{BASE_URL}/predict", json={"history": records})

    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(call) for _ in range(N)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]

    for resp in responses:
        assert resp.status_code == 200, f"Batch request failed: {resp.text}"
        data = resp.json()
        assert len(data["forecast_dates"]) == 7
        assert len(data["quantiles"]) == 7