from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path

config_path = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "config.json"
)

print("Found the config file:", config_path.exists())

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

print("Testing Azure ML workspace access...")

workspace = ml_client.workspaces.get(
    ml_client.workspace_name
)

print("Connected to Azure ML successfully!")
print("Workspace:", workspace.name)
print("Resource group:", workspace.resource_group)