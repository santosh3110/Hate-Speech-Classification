import os
import sys
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.utils.common import clean_single_text
from hate_speech_classifier.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.pusher_config = config.get_model_pusher_config()
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.model_filename = self.pusher_config.model_name
        self.bucket_name = self.pusher_config.bucket_name

    def get_best_model_from_gcs(self):
        try:
            os.makedirs(self.model_path, exist_ok=True)
            command = f"gsutil cp gs://{self.bucket_name}/{self.model_filename} {self.model_path}/"
            os.system(command)
            return os.path.join(self.model_path, self.model_filename)
        except Exception as e:
            raise CustomException(e, sys)

    def load_tokenizer(self):
        try:
            tokenizer_path = os.path.join("artifacts", "split", "tokenizer.pkl")
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            return tokenizer
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, text: str) -> dict:
        try:
            model_file = self.get_best_model_from_gcs()
            model = load_model(model_file)
            tokenizer = self.load_tokenizer()

            # Clean + tokenize
            clean_text = clean_single_text(text)
            sequence = tokenizer.texts_to_sequences([clean_text])
            padded = pad_sequences(sequence, maxlen=100)

            # Predict
            probability = float(model.predict(padded)[0][0])
            label = "Hate" if probability >= 0.4 else "No Hate"

            return {
                "label": label,
                "probability": probability
            }

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, text: str) -> str:
        logging.info("Running prediction pipeline")
        return self.predict(text)
