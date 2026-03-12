#!/bin/bash
set -e

ENGINE_PATH="models/tft_model.trt"
ONNX_PATH="models/tft_model.onnx"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "TRT engine not found at $ENGINE_PATH."

    if [ ! -f "$ONNX_PATH" ]; then
        echo "ERROR: ONNX model not found at $ONNX_PATH. Cannot build engine." >&2
        exit 1
    fi

    echo "Building TRT engine from $ONNX_PATH — this may take several minutes..."
    python -c "
from build_trt_engine import build_engine
build_engine('$ONNX_PATH', '$ENGINE_PATH', fp16=True)
"
    echo "Engine build complete."
else
    echo "TRT engine found at $ENGINE_PATH. Skipping build."
fi

exec python serve_ray.py