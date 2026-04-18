"""
Register the current build in the MLflow Model Registry.

Finds the latest run in experiment 'solar_flux_tft' (or the run
passed via --run-id), attaches ONNX/TRT artifacts to it if they
are present locally (after being fetched from S3 for the Docker
build), then creates a registered model version tagged with the
git SHA and ECR image URI.

Requires: MLFLOW_TRACKING_URI env var pointing at your MLflow server.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME  = "solar_flux_tft"
REGISTERED_MODEL = "solar-flux-tft"

_EXTRA_ARTIFACTS = [
    ("models/tft_model.onnx",      "onnx"),
    ("models/tft_model.trt",       "trt"),
    ("models/tft_model_int8.trt",  "trt"),
    ("models/calib.cache",         "trt"),
]


def _latest_run_id() -> str:
    c = MlflowClient()
    exp = c.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"Experiment {EXPERIMENT_NAME!r} not found in MLflow.")
    runs = c.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment {EXPERIMENT_NAME!r}.")
    return runs[0].info.run_id


def register(run_id: str, tag: str, image_uri: str) -> None:
    client = MlflowClient()

    # Attach ONNX/TRT artifacts to the training run (present after S3 fetch)
    for local_path, artifact_subdir in _EXTRA_ARTIFACTS:
        if Path(local_path).exists():
            size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            print(f"Logging {local_path} ({size_mb:.1f} MB) → run artifacts/{artifact_subdir}/")
            client.log_artifact(run_id, local_path, artifact_path=artifact_subdir)

    # Register the checkpoint as a new model version
    model_uri = f"runs:/{run_id}/checkpoints"
    print(f"Registering {model_uri} as '{REGISTERED_MODEL}'...")
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL)

    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "git_sha",       tag)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "image_uri",     image_uri)
    client.set_model_version_tag(
        REGISTERED_MODEL, mv.version, "registered_at",
        datetime.now(timezone.utc).isoformat(),
    )

    print(f"Registered '{REGISTERED_MODEL}' version {mv.version} (run_id={run_id}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",    default=None,  help="MLflow run ID (default: latest in experiment)")
    parser.add_argument("--tag",       required=True, help="Git SHA")
    parser.add_argument("--image-uri", required=True, help="ECR image URI")
    args = parser.parse_args()

    run_id = args.run_id or _latest_run_id()
    print(f"Using MLflow run: {run_id}")
    register(run_id, args.tag, args.image_uri)