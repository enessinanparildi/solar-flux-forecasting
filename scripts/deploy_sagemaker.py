"""
Deploy or update a SageMaker real-time endpoint.

Usage (full rollout):
    python scripts/deploy_sagemaker.py \
        --image-uri 123456789.dkr.ecr.us-east-1.amazonaws.com/solar-flux-tft:abc123 \
        --endpoint-name solar-flux-tft \
        --role-arn arn:aws:iam::123456789:role/SageMakerExecutionRole \
        --tag abc123

Usage (canary — route 10 % to new model, keep 90 % on current):
    python scripts/deploy_sagemaker.py ... --canary-weight 10
"""

import argparse
import boto3
from botocore.exceptions import ClientError

INSTANCE_TYPE = "ml.g4dn.xlarge"  # 1x T4 GPU — change to ml.g5.xlarge for A10G
INITIAL_INSTANCE_COUNT = 1


def endpoint_exists(sm, endpoint_name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def get_current_variant(sm, endpoint_name: str) -> tuple[str | None, str | None]:
    """Return (model_name, instance_type) of the live primary variant."""
    try:
        ep = sm.describe_endpoint(EndpointName=endpoint_name)
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        v = cfg["ProductionVariants"][0]
        return v["ModelName"], v["InstanceType"]
    except ClientError:
        return None, None


def _variant(name: str, model_name: str, weight: int, instance_type: str) -> dict:
    return {
        "VariantName": name,
        "ModelName": model_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
        "InitialVariantWeight": weight,
        "ContainerStartupHealthCheckTimeoutInSeconds": 600,
    }


def deploy(
    image_uri: str,
    endpoint_name: str,
    role_arn: str,
    tag: str,
    canary_weight: int = 0,
) -> None:
    sm = boto3.client("sagemaker")

    new_model_name = f"{endpoint_name}-{tag}"
    config_name    = f"{endpoint_name}-{tag}"

    print(f"Creating model: {new_model_name}")
    sm.create_model(
        ModelName=new_model_name,
        PrimaryContainer={"Image": image_uri},
        ExecutionRoleArn=role_arn,
    )

    exists = endpoint_exists(sm, endpoint_name)
    current_model, current_instance = get_current_variant(sm, endpoint_name) if exists else (None, None)
    instance = current_instance or INSTANCE_TYPE

    if canary_weight > 0 and exists and current_model:
        # A/B: new canary variant + keep existing primary variant
        print(
            f"Creating A/B config: canary={canary_weight}% (new), "
            f"primary={100 - canary_weight}% (current={current_model})"
        )
        variants = [
            _variant("canary",  new_model_name,  canary_weight,           instance),
            _variant("primary", current_model,   100 - canary_weight,     instance),
        ]
    else:
        # Full rollout — single variant
        if canary_weight > 0 and not exists:
            print("No existing endpoint — ignoring --canary-weight, doing full deploy.")
        variants = [_variant("primary", new_model_name, 1, instance)]

    print(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=variants)

    if exists:
        print(f"Updating existing endpoint: {endpoint_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    print("Waiting for endpoint to be InService (engine build may take a few minutes)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},  # up to 20 min
    )
    print(f"Endpoint {endpoint_name} is InService.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri",     required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn",      required=True)
    parser.add_argument("--tag",           required=True)
    parser.add_argument(
        "--canary-weight",
        type=int,
        default=0,
        metavar="PCT",
        help="Traffic %% to route to the new model (0 = full rollout). "
             "Ignored on first deploy.",
    )
    args = parser.parse_args()

    deploy(args.image_uri, args.endpoint_name, args.role_arn, args.tag, args.canary_weight)