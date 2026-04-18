"""
Validate model artifacts before deployment. No GPU required.

Checks file presence and size sanity for committed artifacts.
ONNX / TRT engines are gitignored and fetched from S3 separately —
their absence is a warning, not a failure, in pure checkout CI.
"""
import argparse
import os
import sys

# Minimum expected sizes
_MIN_SIZES = {
    "dataset_params.pt":  10 * 1024,          # 10 KB
    "calib.cache":         1 * 1024,           # 1 KB
    "tft_model.onnx":      5 * 1024 * 1024,   # 5 MB  (S3-fetched, may be absent)
    "tft_model.trt":       5 * 1024 * 1024,   # 5 MB  (built/fetched, may be absent)
    "tft_model_int8.trt":  5 * 1024 * 1024,
}

_REQUIRED = {"dataset_params.pt"}


def check_file(path: str, min_bytes: int, required: bool) -> bool:
    if not os.path.exists(path):
        tag = "FAIL" if required else "WARN"
        print(f"{tag}: {path} not found")
        return not required
    size = os.path.getsize(path)
    if size < min_bytes:
        print(f"FAIL: {path} too small ({size} B < {min_bytes} B)")
        return False
    print(f"OK  : {path} ({size / 1024:.0f} KB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    d = args.models_dir
    ok = True
    for name, min_bytes in _MIN_SIZES.items():
        required = name in _REQUIRED
        if not check_file(os.path.join(d, name), min_bytes, required):
            ok = False

    if not ok:
        print("\nArtifact validation FAILED.")
        sys.exit(1)

    print("\nArtifact validation passed.")