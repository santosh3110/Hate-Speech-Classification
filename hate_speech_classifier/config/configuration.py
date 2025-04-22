from hate_speech_classifier.utils.common import load_yaml
from hate_speech_classifier.entity.config_entity import (
    DataIngestionConfig, DataTransformationConfig, TokenizerConfig,
    EmbeddingConfig, ModelTrainerConfig, EvaluationConfig
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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(**self.config['data_transformation'])

    def get_tokenizer_config(self) -> TokenizerConfig:
        return TokenizerConfig(**self.config['tokenizer'])

    def get_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(**self.config['embedding'])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(**self.config['model_trainer'])

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(**self.config['evaluation'])
