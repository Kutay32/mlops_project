"""
Promote a model version in MLflow Model Registry to a target stage (e.g., Staging -> Production).

Usage:
  MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/promote_model.py \
      --registered-model-name "UltimateChurnModel" --target-stage Production --version 1
"""

import os
import argparse

from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registered-model-name', type=str, required=True)
    parser.add_argument('--version', type=str, default=None, help='Model version to promote (if omitted, use latest)')
    parser.add_argument('--target-stage', type=str, default='Production')
    parser.add_argument('--archive-existing', action='store_true', help='Archive existing versions in target stage')
    args = parser.parse_args()

    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        print(f"Using MLflow tracking at: {mlflow_uri}")

    client = MlflowClient()

    # Determine version
    if args.version:
        version = args.version
    else:
        # get latest version
        versions = client.get_latest_versions(name=args.registered_model_name)
        if not versions:
            raise ValueError(f"No registered model found with name {args.registered_model_name}")
        version = versions[-1].version

    # Optionally archive existing
    if args.archive_existing:
        existing = client.get_latest_versions(name=args.registered_model_name, stages=[args.target_stage])
        for v in existing:
            client.transition_model_version_stage(
                name=args.registered_model_name,
                version=v.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"Archived version {v.version}")

    # Promote requested version
    client.transition_model_version_stage(
        name=args.registered_model_name,
        version=version,
        stage=args.target_stage,
        archive_existing_versions=args.archive_existing,
    )

    print(f"Promoted model {args.registered_model_name} version {version} to {args.target_stage}")


if __name__ == '__main__':
    main()
