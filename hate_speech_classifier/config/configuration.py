from hate_speech_classifier.utils.common import load_yaml
from hate_speech_classifier.entity.config_entity import (
    DataIngestionConfig, PreprocessingConfig, EmbeddingConfig, ModelConfig, ModelTrainingConfig
)

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config/config.yaml"):
        with open(config_filepath, 'r') as file:
            self.config = load_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        return DataIngestionConfig(
            bucket_name=config["bucket_name"],
            zip_file_name=config["zip_file_name"],
            artifacts_dir=config["artifacts_dir"],
            ingestion_dir=config["ingestion_dir"],
            imbalance_file_name=config["imbalance_file_name"],
            raw_file_name=config["raw_file_name"]
        )

    def get_preprocessing_config(self) -> PreprocessingConfig:
        config = self.config["preprocessing"]

        return PreprocessingConfig(
            cleaned_file_name=config["cleaned_file_name"],
            stopwords=config["stopwords"]
        )
    
    def get_embedding_config(self) -> EmbeddingConfig:
        config = self.config["embeddings"]

        return EmbeddingConfig(
            artifacts_dir=config["artifacts_dir"],
            max_words=config["max_words"],
            max_seq_length=config["max_seq_length"],
            embedding_dim=config["embedding_dim"],
            glove_file=config["glove_file"],
            embedded_matrix_file=config["embedded_matrix_file"],
            tokenizer_file=config["tokenizer_file"]
        )
    
    def get_model_config(self) -> ModelConfig:
        model = self.config['model']
        return ModelConfig(**model)
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        training = self.config['training']
        return ModelTrainingConfig(**training)