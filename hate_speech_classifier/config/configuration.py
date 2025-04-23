from hate_speech_classifier.utils.common import load_yaml
from hate_speech_classifier.entity.config_entity import (
    DataIngestionConfig, PreprocessingConfig
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
