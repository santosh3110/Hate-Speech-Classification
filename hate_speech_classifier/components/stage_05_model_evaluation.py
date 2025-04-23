import os
import sys
import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from hate_speech_classifier.utils.visualize import plot_confusion_matrix
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

            # Load model + data
            model = load_model(self.model_artifacts.trained_model_path)
            X_test = np.load(self.embedding_artifacts.X_test_path)
            y_test = np.load(self.embedding_artifacts.y_test_path)

            # Predict
            preds = model.predict(X_test)
            y_pred = (preds >= self.class_threshold).astype(int)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_hate = report["1"]["f1-score"]
            recall_hate = report["1"]["recall"]
            precision_hate = report["1"]["precision"]

            metrics = {
                "accuracy": acc,
                "f1_hate": f1_hate,
                "recall_hate": recall_hate,
                "precision_hate": precision_hate,
                "classification_report": report
            }

            # Save metrics
            with open(self.config.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Evaluation metrics saved to {self.config.metrics_file}")

            # Confusion matrix
            cm_plot_path = self.config.metrics_file.replace("evaluation_metrics.json", "confusion_matrix.png")
            plot_confusion_matrix(y_test, y_pred, save_path=cm_plot_path)
            logging.info(f"Confusion matrix saved to {cm_plot_path}")

            # Return the score that decides the best model
            logging.info(f"F1-score for hate class: {f1_hate}")
            return f1_hate

        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self) -> bool:
        try:
            logging.info("Initiating Model Evaluation...")
            current_f1_hate = self.evaluate()
            prev_best_f1 = load_best_accuracy(self.config.best_accuracy_file)

            # First time or improved
            if prev_best_f1 is None or current_f1_hate > prev_best_f1:
                save_best_accuracy(self.config.best_accuracy_file, current_f1_hate)
                logging.info("New model is better (based on F1-hate). Accepting.")
                return True
            else:
                logging.info("Older model is better. Keeping the previous one.")
                return False

        except Exception as e:
            raise CustomException(e, sys)
