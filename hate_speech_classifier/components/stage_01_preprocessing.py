import os
import sys
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

from hate_speech_classifier.utils.common import save_csv
from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import PreprocessingConfig
from hate_speech_classifier.entity.artifact_entity import (
    DataIngestionArtifacts,
    PreprocessingArtifacts,
)

nltk.download("stopwords")
stemmer = nltk.stem.PorterStemmer()


class Preprocessing:
    def __init__(self, config: PreprocessingConfig, ingestion_artifacts: DataIngestionArtifacts):
        self.config = config
        self.ingestion_artifacts = ingestion_artifacts

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"&lt;.*?>+", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\w*\d\w*", "", text)
        words = text.split()
        words = [word for word in words if word not in stopwords.words(self.config.stopwords)]
        words = [stemmer.stem(word) for word in words]
        return " ".join(words)

    def initiate(self) -> PreprocessingArtifacts:
        try:
            logging.info("Starting preprocessing stage...")

            df = pd.read_csv(self.ingestion_artifacts.raw_data_file_path)
            df = df[(df["label"] == 0) | (df["label"] == 1)]
            df = df.drop_duplicates()
            df["clean_text"] = df["text"].apply(self.clean_text)

            cleaned_data_path = os.path.join(
                os.path.dirname(self.ingestion_artifacts.raw_data_file_path),
                self.config.cleaned_file_name,
            )
            save_csv(df, cleaned_data_path)

            logging.info(f"Preprocessing complete. Cleaned data saved at: {cleaned_data_path}")
            return PreprocessingArtifacts(cleaned_data_path=cleaned_data_path)

        except Exception as e:
            raise CustomException(e, sys)
