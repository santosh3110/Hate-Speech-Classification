from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str

@dataclass
class PreprocessingArtifacts:
    cleaned_data_path: str

@dataclass
class EmbeddingArtifacts:
    tokenizer_path: str
    embedding_matrix_path: str
    X_train_path: str
    X_test_path: str
    y_train_path: str
    y_test_path: str

@dataclass
class ModelTrainingArtifacts:
    trained_model_path: str
    history_path: str

