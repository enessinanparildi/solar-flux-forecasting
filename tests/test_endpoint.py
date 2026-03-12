import os
import subprocess
import sys
import time

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Heavy GPU/TRT deps — only available on machines with CUDA + TensorRT
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.utils.benchmark as benchmark
    from pytorch_forecasting import TemporalFusionTransformer
    from tft_model import read_csv, get_dataloaders
    from build_trt_engine import load_engine, create_context, _run_inference_sync
    import tensorrt as trt
    HAS_GPU_DEPS = torch.cuda.is_available()
except ImportError:
    HAS_GPU_DEPS = False

requires_gpu = pytest.mark.skipif(not HAS_GPU_DEPS, reason="GPU/TRT deps not available")

# ---------------------------------------------------------------------------
# Paths — overridable via env vars so CI and local both work
# ---------------------------------------------------------------------------
_BASE = os.environ.get("PERCEPTIVESPACE_DIR", "D:/perceptivespace")
CHECKPOINT_PATH  = os.environ.get("CHECKPOINT_PATH",  f"{_BASE}/lightning_logs/version_171/checkpoints/epoch=29-step=2850.ckpt")
ENGINE_PATH      = os.environ.get("ENGINE_PATH",      f"{_BASE}/models/tft_model.trt")
ENGINE_INT8_PATH = os.environ.get("ENGINE_INT8_PATH", f"{_BASE}/models/tft_model_int8.trt")

DAILY_RECORD_FIELDS = [
    "yy", "mm", "dd", "ND", "Cp", "C9", "Obs_F10_7",
    "Kp_0_3", "Kp_3_6", "Kp_6_9", "Kp_9_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "Ap_0_3", "Ap_3_6", "Ap_6_9", "Ap_9_12",
    "Ap_12_15", "Ap_15_18", "Ap_18_21", "Ap_21_24",
]

N_WARMUP = 5
N_RUNS   = 300


def df_to_records(df, n_rows=67):
    """Convert the last n_rows of a dataframe to DailyRecord dicts."""
    rows = df.tail(n_rows)
    records = []
    for _, row in rows.iterrows():
        records.append({field: float(row[field]) for field in DAILY_RECORD_FIELDS})
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server():
    if not HAS_GPU_DEPS:
        pytest.skip("GPU/TRT deps not available")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serve:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=_BASE,
    )
    start = time.time()
    while time.time() - start < 100:
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Server did not start in time")

    yield proc

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def records():
    _, _, test_df, _ = read_csv()
    return df_to_records(test_df, n_rows=67)


@pytest.fixture(scope="module")
def benchmark_batch():
    """A single real batch from the test dataloader, moved to CUDA."""
    _, _, test_dataloader, _ = get_dataloaders()
    batch_x, _ = next(iter(test_dataloader))
    return {k: v.cuda() for k, v in batch_x.items()}


@pytest.fixture(scope="module")
def pytorch_model():
    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval().cuda()
    return model


@pytest.fixture(scope="module")
def compiled_model(pytorch_model, benchmark_batch):
    torch._dynamo.config.capture_scalar_outputs = True
    model = torch.compile(pytorch_model, backend="cudagraphs")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(benchmark_batch)
    torch.cuda.synchronize()
    return model


def _load_trt_inputs(engine_path: str, benchmark_batch: dict) -> tuple:
    engine = load_engine(engine_path)
    context = create_context(engine)
    input_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    ]
    inputs = {name: benchmark_batch[name] for name in input_names}
    return context, inputs


@pytest.fixture(scope="module")
def trt_engine_and_inputs(benchmark_batch):
    return _load_trt_inputs(ENGINE_PATH, benchmark_batch)


@pytest.fixture(scope="module")
def trt_int8_engine_and_inputs(benchmark_batch):
    if not os.path.exists(ENGINE_INT8_PATH):
        pytest.skip(f"INT8 engine not found at {ENGINE_INT8_PATH}")
    return _load_trt_inputs(ENGINE_INT8_PATH, benchmark_batch)


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

def test_predict(server, records):
    resp = requests.post(f"{BASE_URL}/predict", json={"history": records})
    assert resp.status_code == 200, f"Server error: {resp.text}"

    data = resp.json()
    assert len(data["forecast_dates"]) == 7
    assert len(data["quantiles"]) == 7
    assert all(len(q) == 7 for q in data["quantiles"])


def test_too_few_records(server):
    resp = requests.post(f"{BASE_URL}/predict", json={"history": [
        {field: 0.0 for field in DAILY_RECORD_FIELDS} for _ in range(5)
    ]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Latency benchmark: TensorRT vs PyTorch checkpoint
# ---------------------------------------------------------------------------

@requires_gpu
def test_latency_benchmark(pytorch_model, compiled_model, trt_engine_and_inputs, trt_int8_engine_and_inputs, benchmark_batch):
    context_fp16, trt_fp16_inputs = trt_engine_and_inputs
    context_int8, trt_int8_inputs = trt_int8_engine_and_inputs

    trt_fp16_timer = benchmark.Timer(
        stmt="_run_inference_sync(context, inputs)",
        globals={"_run_inference_sync": _run_inference_sync, "context": context_fp16, "inputs": trt_fp16_inputs},
        label="Inference latency",
        sub_label="TensorRT FP16",
        description=f"{N_RUNS} runs",
    )

    trt_int8_timer = benchmark.Timer(
        stmt="_run_inference_sync(context, inputs)",
        globals={"_run_inference_sync": _run_inference_sync, "context": context_int8, "inputs": trt_int8_inputs},
        label="Inference latency",
        sub_label="TensorRT INT8",
        description=f"{N_RUNS} runs",
    )

    pt_timer = benchmark.Timer(
        stmt="""
with torch.no_grad():
    model(batch)
torch.cuda.synchronize()
""",
        globals={"model": pytorch_model, "batch": benchmark_batch, "torch": torch},
        label="Inference latency",
        sub_label="PyTorch eager",
        description=f"{N_RUNS} runs",
    )

    pt_compiled_timer = benchmark.Timer(
        stmt="""
with torch.no_grad():
    model(batch)
torch.cuda.synchronize()
""",
        globals={"model": compiled_model, "batch": benchmark_batch, "torch": torch},
        label="Inference latency",
        sub_label="torch.compile",
        description=f"{N_RUNS} runs",
    )

    trt_fp16_result    = trt_fp16_timer.timeit(N_RUNS)
    trt_int8_result    = trt_int8_timer.timeit(N_RUNS)
    pt_result          = pt_timer.timeit(N_RUNS)
    pt_compiled_result = pt_compiled_timer.timeit(N_RUNS)

    compare = benchmark.Compare([trt_fp16_result, trt_int8_result, pt_compiled_result, pt_result])
    compare.print()

    baseline = pt_result.mean
    print(
        f"\nSpeedups vs PyTorch eager ({baseline * 1e3:.1f} ms):"
        f"\n  TRT FP16:      {baseline / trt_fp16_result.mean:.2f}x  ({trt_fp16_result.mean * 1e3:.1f} ms)"
        f"\n  TRT INT8:      {baseline / trt_int8_result.mean:.2f}x  ({trt_int8_result.mean * 1e3:.1f} ms)"
        f"\n  torch.compile: {baseline / pt_compiled_result.mean:.2f}x  ({pt_compiled_result.mean * 1e3:.1f} ms)"
        f"\n  INT8 vs FP16:  {trt_fp16_result.mean / trt_int8_result.mean:.2f}x"
    )

    assert trt_fp16_result.mean < pt_result.mean, (
        f"Expected TRT FP16 to be faster than PyTorch eager: "
        f"FP16={trt_fp16_result.mean*1e3:.1f}ms PT={pt_result.mean*1e3:.1f}ms"
    )
    assert trt_int8_result.mean < pt_result.mean, (
        f"Expected TRT INT8 to be faster than PyTorch eager: "
        f"INT8={trt_int8_result.mean*1e3:.1f}ms PT={pt_result.mean*1e3:.1f}ms"
    )