
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config" / "config.json"

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

print("Registered data assets:")

assets = list(ml_client.data.list())

if not assets:
    print("No registered data assets found.")
else:
    for asset in assets:
        print(
            f"{asset.name} | "
            f"version={asset.version} | "
            f"type={asset.type}"
        )
