
from pathlib import Path

from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

from azure.ai.ml.constants import AssetTypes


# Resolve paths
src_dir = Path(__file__).resolve().parent
project_root = src_dir.parent
config_path = project_root / "config" / "config.json"

print("Found the config file:", config_path.exists())

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")


ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

from pathlib import Path
from azure.ai.ml import load_component

src_dir = Path(__file__).resolve().parent

loaded_component_prep = load_component(
    source=str(src_dir / "prep_data.yml")
)
loaded_component_train = load_component(
    source=str(src_dir / "train_data.yml")
)
loaded_component_eval = load_component(
    source=str(src_dir / "evaluate.yml")
)

# Retrieve the registered dataset
registered_data_asset = ml_client.data.get(
    name="stock-market-data",
    version="1",
)

@pipeline()
def stock_pipeline(pipeline_job_input):
    prep_data = loaded_component_prep(
        input_data=pipeline_job_input
    )

    train_model = loaded_component_train(
        input_data=prep_data.outputs.output_data
    )

    eval_model = loaded_component_eval(
        input_data=prep_data.outputs.output_data
    )

    return {
        "prepared_data": prep_data.outputs.output_data,
        "trained_model": train_model.outputs.output_data,
        "evaluation_output": eval_model.outputs.output_data,
    }


pipeline_job = stock_pipeline(
    pipeline_job_input=Input(
        type=AssetTypes.URI_FILE,
        path="azureml:stock-market-data:1",
    )
)

pipeline_job.settings.default_compute = "thatobilankulu012"

submitted_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="stock-trading-pipeline",
)

print(f"Submitted pipeline job: {submitted_job.name}")

print(f"Submitted pipeline job: {submitted_job.name}")