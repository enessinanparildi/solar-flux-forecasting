import asyncio
import os
import tensorrt as trt
import numpy as np
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "models/tft_model.onnx"
ENGINE_PATH = "models/tft_model.trt"


class TFTCalibrator(trt.IInt8MinMaxCalibrator):
    """INT8 calibrator for the TFT model.

    Parameters
    ----------
    calibration_data:
        List of input dicts, each mapping tensor name -> CPU torch.Tensor.
        Aim for 500-1000 representative historical windows.
    cache_path:
        Path to read/write the calibration cache.  On subsequent builds the
        cache is reused so profiling only runs once.
    """

    def __init__(self, calibration_data: list[dict[str, torch.Tensor]], cache_path: str) -> None:
        super().__init__()
        self._data = calibration_data
        self._idx = 0
        self._cache_path = cache_path
        # Pre-allocate persistent GPU buffers sized from the first sample.
        # Skipped when calibration_data is empty and an existing cache is used instead.
        self._gpu_bufs: dict[str, torch.Tensor] = (
            {name: torch.zeros_like(tensor, device="cuda") for name, tensor in calibration_data[0].items()}
            if calibration_data
            else {}
        )

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: list[str]):
        if self._idx >= len(self._data):
            return None
        batch = self._data[self._idx]
        self._idx += 1
        for name in names:
            self._gpu_bufs[name].copy_(batch[name])
        return [self._gpu_bufs[name].data_ptr() for name in names]

    def read_calibration_cache(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache: bytes) -> None:
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        with open(self._cache_path, "wb") as f:
            f.write(cache)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = True,
    calibration_data: list[dict[str, torch.Tensor]] | None = None,
    calib_cache: str = "models/calib.cache",
) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This GPU does not support fast INT8 inference")
        if calibration_data is None and not os.path.exists(calib_cache):
            raise ValueError("INT8 requires calibration_data or an existing calib_cache file")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = TFTCalibrator(calibration_data or [], calib_cache)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Dynamic shape profile — adjust min/opt/max to match your expected batch sizes
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        # Replace dynamic dim (batch) with concrete values
        min_shape = [1] + list(shape[1:])
        opt_shape = [64] + list(shape[1:])
        max_shape = [256] + list(shape[1:])
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    print("Building TensorRT engine — this may take a few minutes...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"Engine saved to {engine_path}")


def load_engine(engine_path: str) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def create_context(engine: trt.ICudaEngine) -> trt.IExecutionContext:
    """Create an execution context once — reuse it across inference calls."""
    return engine.create_execution_context()


def _run_inference_sync(context: trt.IExecutionContext, inputs: dict) -> np.ndarray:
    engine = context.engine

    # Set input shapes for dynamic batch
    for name, tensor in inputs.items():
        context.set_input_shape(name, tensor.shape)

    # Allocate GPU buffers
    bindings = {}
    for name, tensor in inputs.items():
        gpu_tensor = tensor.contiguous().cuda()
        bindings[name] = gpu_tensor

    # Allocate output buffers
    output_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    ]
    outputs = {}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        buf = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
        bindings[name] = buf
        outputs[name] = buf

    # Bind all tensors and run
    for name, tensor in bindings.items():
        context.set_tensor_address(name, tensor.data_ptr())

    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return outputs["output"].cpu().numpy()


async def run_inference(context: trt.IExecutionContext, inputs: dict) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_inference_sync, context, inputs)


# ---------------------------------------------------------------------------
# Pipeline parallelism — split model across two GPUs
# ---------------------------------------------------------------------------

def build_engine_on_device(
    onnx_path: str,
    engine_path: str,
    device_id: int = 0,
    fp16: bool = True,
    int8: bool = False,
    calibration_data: list[dict[str, torch.Tensor]] | None = None,
    calib_cache: str = "models/calib.cache",
) -> None:
    """Build a TRT engine pinned to a specific GPU device."""
    torch.cuda.set_device(device_id)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This GPU does not support fast INT8 inference")
        if calibration_data is None and not os.path.exists(calib_cache):
            raise ValueError("INT8 requires calibration_data or an existing calib_cache file")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = TFTCalibrator(calibration_data or [], calib_cache)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        min_shape = [1] + list(shape[1:])
        opt_shape = [64] + list(shape[1:])
        max_shape = [256] + list(shape[1:])
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    print(f"Building TRT engine on GPU {device_id} — this may take a few minutes...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"Engine saved to {engine_path}")


def load_engine_on_device(engine_path: str, device_id: int = 0) -> trt.ICudaEngine:
    """Deserialise a TRT engine into the specified GPU device's context."""
    torch.cuda.set_device(device_id)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def _run_stage_sync(
    context: trt.IExecutionContext, inputs: dict, device_id: int
) -> dict[str, torch.Tensor]:
    """Run one pipeline stage on `device_id`. Returns output tensors still on that device."""
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    stage_engine = context.engine

    for name, tensor in inputs.items():
        context.set_input_shape(name, tensor.shape)

    bindings: dict[str, torch.Tensor] = {}
    for name, tensor in inputs.items():
        bindings[name] = tensor.contiguous().to(device)

    output_names = [
        stage_engine.get_tensor_name(i)
        for i in range(stage_engine.num_io_tensors)
        if stage_engine.get_tensor_mode(stage_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    ]
    outputs: dict[str, torch.Tensor] = {}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        buf = torch.empty(tuple(shape), dtype=torch.float32, device=device)
        bindings[name] = buf
        outputs[name] = buf

    for name, tensor in bindings.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.current_stream(device)
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize(device)

    return outputs


class PipelinedInference:
    """Two-stage pipeline parallelism across two GPUs.

    Stage 0 (encoder_device): runs the encoder TRT engine.
    Stage 1 (decoder_device): runs the decoder TRT engine.

    Activations are transferred between devices via PyTorch peer copy
    (zero host staging when NVLink / peer access is available).

    Parameters
    ----------
    encoder_context, decoder_context:
        Execution contexts for the two split engines.
    encoder_device, decoder_device:
        CUDA device indices for each stage.
    activation_map:
        Dict mapping encoder output tensor names to the corresponding decoder
        input tensor names.  If an encoder output name matches the decoder
        input name exactly, you can omit it from this dict.
    output_name:
        Name of the final output tensor produced by the decoder engine.
    """

    def __init__(
        self,
        encoder_context: trt.IExecutionContext,
        decoder_context: trt.IExecutionContext,
        encoder_device: int = 0,
        decoder_device: int = 1,
        activation_map: dict[str, str] | None = None,
        output_name: str = "output",
    ) -> None:
        self.encoder_ctx = encoder_context
        self.decoder_ctx = decoder_context
        self.encoder_device = encoder_device
        self.decoder_device = decoder_device
        self.activation_map = activation_map or {}
        self.output_name = output_name
        self._check_peer_access()

    def _check_peer_access(self) -> None:
        """Warn early if NVLink / P2P is unavailable (transfers will go via host)."""
        can_peer = torch.cuda.can_device_access_peer(self.decoder_device, self.encoder_device)
        if can_peer:
            print(
                f"P2P access available: GPU {self.encoder_device} -> GPU {self.decoder_device}"
            )
        else:
            print(
                f"Warning: no P2P access between GPU {self.encoder_device} and "
                f"GPU {self.decoder_device}. Activations will stage through host memory."
            )

    def _run_sync(
        self,
        encoder_inputs: dict[str, torch.Tensor],
        extra_decoder_inputs: dict[str, torch.Tensor] | None = None,
    ) -> np.ndarray:
        # --- Stage 0: encoder ---
        enc_outputs = _run_stage_sync(self.encoder_ctx, encoder_inputs, self.encoder_device)

        # Transfer activations to decoder device (P2P copy when available)
        decoder_inputs: dict[str, torch.Tensor] = {}
        for enc_name, tensor in enc_outputs.items():
            dec_name = self.activation_map.get(enc_name, enc_name)
            decoder_inputs[dec_name] = tensor.to(f"cuda:{self.decoder_device}")

        # Merge decoder-only inputs (e.g. decoder time features, static context)
        if extra_decoder_inputs:
            decoder_inputs.update(extra_decoder_inputs)

        # --- Stage 1: decoder ---
        dec_outputs = _run_stage_sync(self.decoder_ctx, decoder_inputs, self.decoder_device)

        return dec_outputs[self.output_name].cpu().numpy()

    async def run(
        self,
        encoder_inputs: dict[str, torch.Tensor],
        extra_decoder_inputs: dict[str, torch.Tensor] | None = None,
    ) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync, encoder_inputs, extra_decoder_inputs)


# ---------------------------------------------------------------------------
# INT8 calibration helpers
# ---------------------------------------------------------------------------

def prepare_calibration_data(n_samples: int = 500) -> list[dict[str, torch.Tensor]]:
    """Build calibration data from the training split.

    Reuses read_csv / wrap_datasets from tft_model.py so the preprocessing
    (cycle features, normalisation, etc.) is identical to training.

    Each returned dict is a single-sample (batch_size=1) slice matching the
    ONNX model's input names exactly (encoder_cat, encoder_cont, …).

    Parameters
    ----------
    n_samples:
        Number of individual windows to collect.  500 is a good default —
        enough to cover the full range of solar-cycle phases without being
        prohibitively slow.
    """
    from tft_model import read_csv, wrap_datasets

    print(f"Loading training data for INT8 calibration ({n_samples} samples)...")
    train_df, valid_df, test_df, train_df_all = read_csv()
    train_dl, _, _, _ = wrap_datasets(train_df, valid_df, test_df, train_df_all)

    max_prediction_length = 7   # must match TimeSeriesDataSet config in tft_model.py
    max_encoder_length = 60     # must match TimeSeriesDataSet config in tft_model.py

    calib_data: list[dict[str, torch.Tensor]] = []
    for x, _ in train_dl:
        batch_size = next(iter(x.values())).shape[0]
        for i in range(batch_size):
            # Skip samples with truncated sequences.  The ONNX model's Where
            # ops build attention masks of shape (encoder_length + decoder_length)
            # so every calibration sample must have both at their maximum values.
            if x["decoder_lengths"][i].item() < max_prediction_length:
                continue
            if x["encoder_lengths"][i].item() < max_encoder_length:
                continue
            # Slice to batch_size=1 so TFTCalibrator.get_batch_size() == 1 holds
            calib_data.append({k: v[i : i + 1].cpu() for k, v in x.items()})
            if len(calib_data) >= n_samples:
                print(f"Collected {len(calib_data)} calibration samples.")
                return calib_data

    print(f"Collected {len(calib_data)} calibration samples (dataloader exhausted).")
    return calib_data


def run_int8_calibration(
    onnx_path: str = ONNX_PATH,
    engine_path: str = ENGINE_PATH,
    calib_cache: str = "models/calib.cache",
    n_samples: int = 500,
    fp16: bool = True,
) -> None:
    """End-to-end helper: load data → calibrate → build INT8 engine.

    On the first run this profiles activation ranges and writes calib_cache.
    On subsequent runs the cache is reused — calibration_data is not needed.
    """
    calib_data = None
    if not os.path.exists(calib_cache):
        calib_data = prepare_calibration_data(n_samples)

    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=fp16,
        int8=True,
        calibration_data=calib_data,
        calib_cache=calib_cache,
    )


if __name__ == "__main__":
    # --- INT8 calibration path ---
    # First run: collects 500 training windows, profiles activations, writes cache.
    # Subsequent runs: cache is reused, data loading is skipped.

    run_int8 = True

    if run_int8:
        run_int8_calibration(
            onnx_path=ONNX_PATH,
            engine_path="models/tft_model_int8.trt",
            calib_cache="models/calib.cache",
            n_samples=500,
        )
    # --- Single-GPU path (unchanged) ---
    else:
        build_engine(ONNX_PATH, ENGINE_PATH, fp16=True)
        engine = load_engine(ENGINE_PATH)
        print(f"Engine loaded. Inputs: {engine.num_io_tensors - 1}, Output: 1")


    # --- Two-GPU pipeline path ---
    # Prerequisites:
    #   1. Export two ONNX files: tft_encoder.onnx and tft_decoder.onnx
    #      from your PyTorch model before calling this.
    #   2. Identify the tensor names at the encoder/decoder boundary
    #      (e.g. "encoder_hidden", "encoder_cell" from the LSTM).
    #
    # build_engine_on_device("models/tft_encoder.onnx", "models/tft_encoder.trt", device_id=0)
    # build_engine_on_device("models/tft_decoder.onnx", "models/tft_decoder.trt", device_id=1)
    #
    # enc_engine = load_engine_on_device("models/tft_encoder.trt", device_id=0)
    # dec_engine = load_engine_on_device("models/tft_decoder.trt", device_id=1)
    #
    # pipeline = PipelinedInference(
    #     encoder_context=enc_engine.create_execution_context(),
    #     decoder_context=dec_engine.create_execution_context(),
    #     encoder_device=0,
    #     decoder_device=1,
    #     # Map encoder output names -> decoder input names if they differ:
    #     activation_map={"encoder_hidden": "decoder_hidden", "encoder_cell": "decoder_cell"},
    #     output_name="output",
    # )
    #
    # result = asyncio.run(pipeline.run(encoder_inputs={...}, extra_decoder_inputs={...}))