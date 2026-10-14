
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Project paths
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent
config_path = project_root / "config" / "config.json"
model_path = project_root / "models" / "stock_model.joblib"

# Validate configuration and model
if not config_path.exists():
    raise FileNotFoundError(f"Config not found: {config_path}")

if not model_path.exists():
    # Also check src in case the model was saved there
    model_path = src_dir / "stock_model.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Model file not found. Checked project models/ "
        "and src/: stock_model.joblib"
    )

print(f"Config: {config_path}")
print(f"Model: {model_path}")

# Connect to the configured workspace
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

# Register the local model
model = Model(
    name="stock-forecast-model",
    path=str(model_path),
    type=AssetTypes.CUSTOM_MODEL,
    description="Stock forecasting model serialized with joblib",
)

registered_model = ml_client.models.create_or_update(model)

print("\nModel registered successfully.")
print(f"Name: {registered_model.name}")
print(f"Version: {registered_model.version}")
print(f"Type: {registered_model.type}")
