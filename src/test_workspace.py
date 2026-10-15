from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

credential = AzureCliCredential(
    tenant_id="996f660a-e105-44e8-808b-9de407af4085"
)

ml_client = MLClient.from_config(
    credential=credential,
    path=r"C:\Users\Thato Bilankulu\Documents\stock-trading\config\config.json"
)

workspace = ml_client.workspaces.get("stock-trading-ml")

print("Workspace:", workspace.name)
print("Location:", workspace.location)
print("Resource group:", workspace.resource_group)