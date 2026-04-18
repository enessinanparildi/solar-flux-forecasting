"""
Gate deployment by checking production endpoint health via CloudWatch.

Queries the last 24 h of SageMaker metrics and fails if:
  - Model error rate > MAX_ERROR_RATE
  - p99 ModelLatency > MAX_P99_LATENCY_US

A first-deploy (endpoint does not yet exist) always passes.
ModelLatency is reported in microseconds by SageMaker.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import ClientError

MAX_ERROR_RATE      = 0.05          # 5 %
MAX_P99_LATENCY_US  = 5_000_000     # 5 s in microseconds


def _cw_stat(cw, endpoint_name: str, metric: str, stat: str) -> float | None:
    """Fetch a single scalar from CloudWatch (standard statistic)."""
    now = datetime.now(timezone.utc)
    resp = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric,
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName",  "Value": "primary"},
        ],
        StartTime=now - timedelta(hours=24),
        EndTime=now,
        Period=86_400,
        Statistics=[stat],
    )
    pts = resp.get("Datapoints", [])
    return pts[0].get(stat) if pts else None


def _cw_percentile(cw, endpoint_name: str, metric: str, pct: str) -> float | None:
    """Fetch a percentile statistic (e.g. 'p99') from CloudWatch."""
    now = datetime.now(timezone.utc)
    resp = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric,
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName",  "Value": "primary"},
        ],
        StartTime=now - timedelta(hours=24),
        EndTime=now,
        Period=86_400,
        ExtendedStatistics=[pct],
    )
    pts = resp.get("Datapoints", [])
    return pts[0].get("ExtendedStatistics", {}).get(pct) if pts else None


def check(endpoint_name: str, region: str) -> None:
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint {endpoint_name!r} does not exist — first deploy, passing.")
            return
        raise

    cw = boto3.client("cloudwatch", region_name=region)

    invocations = _cw_stat(cw, endpoint_name, "Invocations", "Sum") or 0
    if invocations == 0:
        print("No traffic in last 24 h — skipping metric gates.")
        return

    errors = _cw_stat(cw, endpoint_name, "ModelErrors", "Sum") or 0
    p99_us = _cw_percentile(cw, endpoint_name, "ModelLatency", "p99")

    error_rate = errors / invocations
    print(
        f"Prod metrics (24 h): invocations={invocations:.0f}, "
        f"error_rate={error_rate:.3f}, p99={p99_us} µs"
    )

    failed = False
    if error_rate > MAX_ERROR_RATE:
        print(f"FAIL: error rate {error_rate:.1%} exceeds threshold {MAX_ERROR_RATE:.1%}")
        failed = True
    if p99_us is not None and p99_us > MAX_P99_LATENCY_US:
        print(f"FAIL: p99 latency {p99_us / 1e6:.2f} s exceeds threshold {MAX_P99_LATENCY_US / 1e6:.1f} s")
        failed = True

    if failed:
        print("Production endpoint is unhealthy — aborting deployment.")
        sys.exit(1)

    print("Production health check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    check(args.endpoint_name, args.region)