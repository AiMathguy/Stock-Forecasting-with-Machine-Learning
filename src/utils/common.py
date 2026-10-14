import os 
from pydantic import json
import yaml
from pathlib import Path
from src import logger
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any
from box.exceptions import BoxValueError, BoxKeyError


@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the yaml file."""

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except (BoxValueError, BoxKeyError) as e:
        logger.error(f"Error occurred while loading yaml file: {path_to_yaml}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories:list)->None:
    """Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of paths to directories to be created.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created at: {path}")
@ensure_annotations
def save_json(path:Path, data:dict)->None:
    """Saves a dictionary as a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
        data (dict): Dictionary to be saved as JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
@ensure_annotations
def load_json(path:Path)->dict:
    """Loads a JSON file and returns it as a dictionary.
    
    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    with open(path, "r") as f:
        data = json.load(f)
        logger.info(f"JSON file loaded from: {path}")
        return data

@ensure_annotations
def save_binary(path:Path, data:Any)->None:
    """Saves data as a binary file using joblib.
    
    Args:
        path (Path): Path to the binary file.
        data (Any): Data to be saved as binary.
    """
    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_binary(path:Path)->Any:
    """Loads a binary file using joblib and returns the data.
    
    Args:
        path (Path): Path to the binary file.
        
    Returns:
        Any: The data loaded from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data