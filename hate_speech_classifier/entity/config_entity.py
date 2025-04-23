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

@dataclass
class EmbeddingConfig:
    max_words: int
    max_seq_length: int
    embedding_dim: int
    glove_file: str
    embedded_matrix_file: str
    tokenizer_file: str

