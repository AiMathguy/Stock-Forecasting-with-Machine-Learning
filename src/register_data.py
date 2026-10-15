from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

project_root = Path(__file__).resolve().parent.parent
data_path = Path(__file__).resolve().parent / "stock_data.csv"

from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

# Resolve project paths
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent

config_path = project_root / "config" / "config.json"
data_path = src_dir / "stock_data.csv"

# Validate paths before connecting

if not config_path.exists():
    raise FileNotFoundError(f"Azure ML config not found: {config_path}")

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print(f"Config: {config_path}")
print(f"Dataset: {data_path}")

# Connect to Azure ML
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

# Register the dataset
stock_data = Data(
    name="stock-market-data",
    version="1",
    description="Historical stock market data for ML forecasting",
    type="uri_file",
    path=str(data_path),
)

registered_data = ml_client.data.create_or_update(stock_data)

print("\nDataset registered successfully!")
print(f"Name: {registered_data.name}")
print(f"Version: {registered_data.version}")
print(f"Asset ID: {registered_data.id}")
