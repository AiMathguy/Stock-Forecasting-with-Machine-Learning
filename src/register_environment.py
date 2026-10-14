
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

config_path = Path(
    r"C:\Users\Thato Bilankulu\Documents\stock-trading\config\config.json"
)

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path)
)
env = Environment(
    name="stock-prediction-env",
    description="Environment for stock prediction inference",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)

registered_env = ml_client.environments.create_or_update(env)

print("Environment:", registered_env.name)
print("Version:", registered_env.version)
