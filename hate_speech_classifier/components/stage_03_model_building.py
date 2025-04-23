import sys
import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import (
    Embedding, Conv1D, MaxPooling1D, Bidirectional,
    LSTM, Dense, Dropout
)
from keras.regularizers import l2

from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import ModelConfig
from hate_speech_classifier.entity.artifact_entity import EmbeddingArtifacts


class ModelBuilder:
    def __init__(self, config: ModelConfig, embedding_artifacts: EmbeddingArtifacts):
        self.config = config
        self.embedding_artifacts = embedding_artifacts

    def build_model(self):
        try:
            logging.info("Model building started...")

            # Load embedding matrix
            embedding_matrix = np.load(self.embedding_artifacts.embedding_matrix_path)

            model = Sequential([
                Embedding(
                    input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=True
                ),
                Conv1D(filters=self.config.conv1d_filters, kernel_size=self.config.kernel_size, activation=self.config.activation),
                MaxPooling1D(pool_size=self.config.pool_size),
                Bidirectional(LSTM(
                    self.config.lstm_units,
                    dropout=self.config.lstm_dropout_rate,
                    recurrent_dropout=self.config.lstm_recurrent_dropout_rate
                )),
                Dropout(self.config.dropout_rate),
                Dense(self.config.dense_units, activation=self.config.activation, kernel_regularizer=l2(self.config.l2_regularization)),
                Dropout(self.config.dense_dropout_rate),
                Dense(1, activation=self.config.final_activation)
            ])

            os.makedirs(os.path.dirname(self.config.model_architecture_file), exist_ok=True)
            with open(self.config.model_architecture_file, "w") as f:
                f.write(model.to_json())

            logging.info(f"Model architecture saved to {self.config.model_architecture_file}")
            return model

        except Exception as e:
            raise CustomException(e, sys)
