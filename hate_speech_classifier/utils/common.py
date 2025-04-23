import yaml
import pandas as pd
import pickle
import json
import numpy as np
from typing import Any
from box import ConfigBox  
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stemmer = nltk.stem.PorterStemmer()

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

def clean_single_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"&lt;.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)



