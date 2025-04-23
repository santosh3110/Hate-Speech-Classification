import yaml
import pandas as pd
import pickle
import json
import numpy as np
from typing import Any
from box import ConfigBox  

def load_yaml(path_to_yaml: str) -> ConfigBox:
    """Loads YAML file and returns ConfigBox for dot-access"""
    with open(path_to_yaml, 'r') as f:
        content = yaml.safe_load(f)
    return ConfigBox(content)

def write_yaml(path: str, data: Any) -> None:
    """Writes data to a YAML file"""
    with open(path, 'w') as file:
        yaml.dump(data, file)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def save_numpy(array, path):
    np.save(path, array)

def save_json(data: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


