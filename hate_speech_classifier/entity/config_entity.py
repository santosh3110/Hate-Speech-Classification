from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionConfig:
    bucket_name: str
    zip_file_name: str
    artifacts_dir: str
    ingestion_dir: str
    imbalance_file_name: str
    raw_file_name: str

@dataclass
class PreprocessingConfig:
    cleaned_file_name: str
    stopwords: str