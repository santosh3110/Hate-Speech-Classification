import os
import sys
import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

from hate_speech_classifier.utils.metrics import save_best_accuracy, load_best_accuracy
from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import ModelEvaluationConfig
from hate_speech_classifier.entity.artifact_entity import ModelTrainingArtifacts, EmbeddingArtifacts


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig,
                 model_artifacts: ModelTrainingArtifacts,
                 embedding_artifacts: EmbeddingArtifacts,
                 class_threshold: float):
        self.config = config
        self.model_artifacts = model_artifacts
        self.embedding_artifacts = embedding_artifacts
        self.class_threshold = class_threshold

    def evaluate(self):
        try:
            logging.info("Evaluating the model...")

            model = load_model(self.model_artifacts.trained_model_path)
            X_test = np.load(self.embedding_artifacts.X_test_path)
            y_test = np.load(self.embedding_artifacts.y_test_path)

            preds = model.predict(X_test)
            y_pred = (preds >= self.class_threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            metrics = {
                "accuracy": acc,
                "classification_report": report
            }

            # Save evaluation metrics
            with open(self.config.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Evaluation metrics saved to {self.config.metrics_file}")
            return acc

        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self) -> bool:
        try:
            current_acc = self.evaluate()
            prev_best_acc = load_best_accuracy(self.config.best_accuracy_file)

            if prev_best_acc is None or current_acc > prev_best_acc:
                save_best_accuracy(self.config.best_accuracy_file, current_acc)
                logging.info("New model is better. Accepting.")
                return True
            else:
                logging.info("Old model is better. Rejecting.")
                return False
        except Exception as e:
            raise CustomException(e, sys)
