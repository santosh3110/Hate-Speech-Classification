import os
import sys

from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import ModelPusherConfig
from hate_speech_classifier.entity.artifact_entity import ModelTrainingArtifacts


class ModelPusher:
    def __init__(self, config: ModelPusherConfig, model_artifacts: ModelTrainingArtifacts):
        self.config = config
        self.model_artifacts = model_artifacts

    def initiate(self):
        try:
            command = f"gsutil cp {self.model_artifacts.trained_model_path} gs://{self.config.bucket_name}/{self.config.model_name}"
            os.system(command)
            logging.info("Model pushed to GCS successfully.")
            return True
        except Exception as e:
            raise CustomException(e, sys)
