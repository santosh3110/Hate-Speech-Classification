import yaml
from typing import Any
from box import ConfigBox  # for dot-access config

def load_yaml(path_to_yaml: str) -> ConfigBox:
    """Loads YAML file and returns ConfigBox for dot-access"""
    with open(path_to_yaml, 'r') as f:
        content = yaml.safe_load(f)
    return ConfigBox(content)

def write_yaml(path: str, data: Any) -> None:
    """Writes data to a YAML file"""
    with open(path, 'w') as file:
        yaml.dump(data, file)
