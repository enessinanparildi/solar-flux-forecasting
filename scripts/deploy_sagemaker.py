"""
Deploy or update a SageMaker real-time endpoint.

Usage:
    python scripts/deploy_sagemaker.py \
        --image-uri 123456789.dkr.ecr.us-east-1.amazonaws.com/solar-flux-tft:abc123 \
        --endpoint-name solar-flux-tft \
        --role-arn arn:aws:iam::123456789:role/SageMakerExecutionRole \
        --tag abc123
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


def deploy(image_uri: str, endpoint_name: str, role_arn: str, tag: str) -> None:
    sm = boto3.client("sagemaker")

    model_name = f"{endpoint_name}-{tag}"
    config_name = f"{endpoint_name}-{tag}"

    print(f"Creating model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri},
        ExecutionRoleArn=role_arn,
    )

    print(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,  # allow engine build time
            }
        ],
    )

    if endpoint_exists(sm, endpoint_name):
        print(f"Updating existing endpoint: {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    print("Waiting for endpoint to be InService (engine build may take a few minutes)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},  # up to 20 min
    )
    print(f"Endpoint {endpoint_name} is InService.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    deploy(args.image_uri, args.endpoint_name, args.role_arn, args.tag)