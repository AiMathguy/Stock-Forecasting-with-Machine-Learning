
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
    Model,
)
from azure.identity import AzureCliCredential

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

ENDPOINT_NAME = "stock-prediction-endpoint"
DEPLOYMENT_NAME = "stock-prediction-blue"
MODEL_NAME = "stock-prediction-model"
ENV_NAME = "stock-prediction-env"

# Latest artifact shown in your output; verify this is the intended model.
MODEL_PATH = (
    SRC_DIR
    / "mlruns"
    / "229241551780100562"
    / "ff2be5984523492b918d2764ba8902ec"
    / "artifacts"
    / "model"
)

SCORING_DIR = SRC_DIR


def main():
    print("Connecting to workspace...")

    credential = AzureCliCredential(
        tenant_id="996f660a-e105-44e8-808b-9de407af4085"
    )

    ml_client = MLClient.from_config(
        credential=credential,
        path=str(PROJECT_ROOT / "config" / "config.json"),
    )

    # Validate local files before sending resources to Azure.
    if not (MODEL_PATH / "MLmodel").is_file():
        raise FileNotFoundError(f"MLflow model not found: {MODEL_PATH}")

    if not (MODEL_PATH / "model.pkl").is_file():
        raise FileNotFoundError(f"Model pickle not found in: {MODEL_PATH}")

    score_file = SCORING_DIR / "score.py"
    if not score_file.is_file():
        raise FileNotFoundError(
            f"Scoring script not found: {score_file}. "
            "Update SCORING_DIR to the directory containing score.py."
        )

    print("Registering MLflow model...")
    model = Model(
        name=MODEL_NAME,
        path=str(MODEL_PATH),
        type="mlflow_model",
        description="Stock prediction scikit-learn model",
    )
    registered_model = ml_client.models.create_or_update(model)
    print(
        f"Registered model: {registered_model.name}:"
        f"{registered_model.version}"
    )

    print("Registering inference environment...")
    environment = Environment(
        name=ENV_NAME,
        conda_file=str(SRC_DIR / "environment.yml"),
        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu22.04:latest"
        ),
    )
    registered_env = ml_client.environments.create_or_update(environment)
    print(
        f"Registered environment: {registered_env.name}:"
        f"{registered_env.version}"
    )

    print(f"Creating/updating endpoint: {ENDPOINT_NAME}")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="REST API for stock prediction",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Creating/updating deployment: {DEPLOYMENT_NAME}")
    deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=registered_model,
    environment=registered_env,
    code_configuration=CodeConfiguration(
        code=str(SCORING_DIR),
        scoring_script="score.py",
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

    ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()

    # Route all endpoint traffic to this deployment.
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Deployment completed.")
    print(
        "Scoring URI:",
        ml_client.online_endpoints.get(ENDPOINT_NAME).scoring_uri,
    )


if __name__ == "__main__":
    main()
