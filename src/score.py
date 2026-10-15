
import json
import logging
import os

import mlflow.pyfunc
import pandas as pd


def init():
    global model

    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    # Azure ML registered the MLflow artifact under model/
    mlflow_model_path = os.path.join(model_dir, "model")

    logging.info("Loading MLflow model from %s", mlflow_model_path)
    model = mlflow.pyfunc.load_model(mlflow_model_path)
    logging.info("MLflow model loaded successfully")


def run(raw_data):
    try:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        # Accept {"data": [...]} or a direct list of feature rows.
        if isinstance(raw_data, dict) and "data" in raw_data:
            data = raw_data["data"]
        else:
            data = raw_data

        if isinstance(data, dict):
            data = pd.DataFrame([data])
        else:
            data = pd.DataFrame(data)

        predictions = model.predict(data)
        logging.info("Request processed successfully")

        return {"predictions": predictions.tolist()}

    except Exception:
        logging.exception("Prediction failed")
        raise
