import os
import sys
import json
import numpy as np
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint

from hate_speech_classifier.utils.common import save_json
from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import ModelTrainingConfig
from hate_speech_classifier.entity.artifact_entity import EmbeddingArtifacts, ModelTrainingArtifacts

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig, embedding_artifacts: EmbeddingArtifacts):
        self.config = config
        self.artifacts = embedding_artifacts

    def load_model(self):
        try:
            with open("artifacts/model/model_architecture.json", 'r') as f:
                model_json = f.read()
            model = model_from_json(model_json)
            logging.info("Model architecture loaded.")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self) -> ModelTrainingArtifacts:
        try:
            logging.info("Starting model training...")

            model = self.load_model()
            model.compile(
                loss=self.config.loss,
                optimizer=self.config.optimizer,
                metrics=self.config.metrics
            )

            # Load train data
            X_train = np.load(self.artifacts.X_train_path)
            y_train = np.load(self.artifacts.y_train_path)

            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.config.patience, restore_best_weights=True),
                ModelCheckpoint(filepath=self.config.model_save_path, monitor='val_loss', save_best_only=True)
            ]

            # Fit model
            history = model.fit(
                X_train,
                y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1
            )

            # Save history
            history_path = self.config.model_save_path.replace(".h5", "_history.json")
            save_json(history.history, history_path)

            logging.info("Model training completed.")
            return ModelTrainingArtifacts(
                trained_model_path=self.config.model_save_path,
                history_path=history_path
            )

        except Exception as e:
            raise CustomException(e, sys)
