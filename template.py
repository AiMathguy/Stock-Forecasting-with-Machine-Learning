import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
list_of_files = [f"{Path(__file__).parent}/.github/workflows/.gitkeep", 
                 f"{Path(__file__).parent}/.gitignore", 
                 f"{Path(__file__).parent}/.vscode/settings.json",
                   f"{Path(__file__).parent}/data/.gitkeep",
                     f"{Path(__file__).parent}/main.py", 
                     f"{Path(__file__).parent}/README.md",
                       f"{Path(__file__).parent}/requirements.txt",
                         f"{Path(__file__).parent}/src/__init__.py", 
                         f"{Path(__file__).parent}/src/template.py",
                           f"{Path(__file__).parent}/utils/__init__.py", 
                         f"{Path(__file__).parent}/src/utils/common.py",
                         f"{Path(__file__).parent}/config/__init__.py",
                         f"{Path(__file__).parent}/config/config.py",
                         f"{Path(__file__).parent}/config/config.yaml"
                         f"{Path(__file__).parent}/tests/__init__.py",
                         f"{Path(__file__).parent}/params/__init__.py",
                         f"{Path(__file__).parent}/params/params.yaml",
                         f"{Path(__file__).parent}/schema.yaml",
                         f"{Path(__file__).parent}/main.py",
                         f"{Path(__file__).parent}/Dockerfile",
                         f"{Path(__file__).parent}/setup.py",
                         f"{Path(__file__).parent}/experiments/exp1.py",

                         ]

for filepath in list_of_files:
    file_dir, file_name = os.path.split(filepath)
    os.makedirs(file_dir, exist_ok=True)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")