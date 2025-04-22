import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "hate_speech_classifier"

list_of_files = [
    f"{project_name}/__init__.py",

    # 🔹 COMPONENTS (pipeline stages)
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/stage_00_data_loading.py",
    f"{project_name}/components/stage_01_preprocessing.py",
    f"{project_name}/components/stage_02_embeddings.py",
    f"{project_name}/components/stage_03_model_building.py",
    f"{project_name}/components/stage_04_model_training.py",
    f"{project_name}/components/stage_05_model_evaluation.py",
    f"{project_name}/components/stage_06_prediction.py",

    # 🔹 CONFIGURATION
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",

    # 🔹 CONSTANTS
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/global_constants.py",

    # 🔹 ENTITY CLASSES
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",

    # 🔹 EXCEPTION HANDLING
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception_handler.py",

    # 🔹 LOGGING
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/log.py",

    # 🔹 PIPELINE ORCHESTRATION
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/train_pipeline.py",
    f"{project_name}/pipeline/predict_pipeline.py",

    # 🔹 UTILITIES
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",
    f"{project_name}/utils/metrics.py",
    f"{project_name}/utils/visualize.py",

    # 🔹 CONFIG FILES
    "config/config.yaml",

    # 🔹 MAIN ENTRYPOINT
    "main.py",               # Train + Evaluate
    "predict.py",            # Single prediction script
    "setup.py",              # pip install -e .

    # 🔹 MISC FILES
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "requirements.txt",
    "README.md",

    # 🔹 DATA, ARTIFACTS, MODELS, LOGS
    "artifacts/.gitkeep",
    "saved_models/.gitkeep",
    "logs/.gitkeep",
    "data/.gitkeep",
    "research/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} already exists")
