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
    artifacts_dir: str
    max_words: int
    max_seq_length: int
    embedding_dim: int
    glove_file: str
    embedded_matrix_file: str
    tokenizer_file: str

@dataclass
class ModelConfig:
    model_architecture_file: str
    conv1d_filters: int
    kernel_size: int
    pool_size: int
    lstm_units: int
    lstm_dropout_rate: float
    lstm_recurrent_dropout_rate: float
    dropout_rate: float
    dense_units: int
    dense_dropout_rate: float
    l2_regularization: float
    activation: str
    final_activation: str

@dataclass
class ModelTrainingConfig:
    loss: str
    optimizer: str
    metrics: list
    model_save_path: str
    epochs: int
    batch_size: int
    validation_split: float
    patience: int
    class_threshold: float

@dataclass
class ModelEvaluationConfig:
    metrics_file: str
    best_accuracy_file: str

@dataclass
class ModelPusherConfig:
    bucket_name: str
    model_name: str

