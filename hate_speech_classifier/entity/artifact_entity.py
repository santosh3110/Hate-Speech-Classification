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
    padded_sequences_path: str
    tokenizer_path: str
    embedding_matrix_path: str
