import os
import sys
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.utils.common import save_numpy, save_pickle
from hate_speech_classifier.entity.config_entity import EmbeddingConfig
from hate_speech_classifier.entity.artifact_entity import PreprocessingArtifacts, EmbeddingArtifacts


class EmbeddingLayer:
    def __init__(self, config: EmbeddingConfig, preprocessing_artifacts: PreprocessingArtifacts):
        self.config = config
        self.artifacts = preprocessing_artifacts

    def initiate(self) -> EmbeddingArtifacts:
        try:
            logging.info("Starting embedding layer creation...")

            df = pd.read_csv(self.artifacts.cleaned_data_path)
            texts = df["clean_text"].fillna("").astype(str).tolist()


            tokenizer = Tokenizer(num_words=self.config.max_words)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.config.max_seq_length)

            # Save padded
            padded_path = self.artifacts.cleaned_data_path.replace("cleaned_data.csv", "padded.npy")
            np.save(padded_path, padded)

            # Save tokenizer
            tokenizer_path = os.path.join(os.path.dirname(padded_path), self.config.tokenizer_file)
            with open(tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)

            # Load GloVe
            glove_index = {}
            with open(self.config.glove_file, encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    glove_index[word] = vector

            # Create embedding matrix
            embedding_matrix = np.zeros((self.config.max_words, self.config.embedding_dim))
            for word, i in tokenizer.word_index.items():
                if i < self.config.max_words:
                    vector = glove_index.get(word)
                    if vector is not None:
                        embedding_matrix[i] = vector

            # Save matrix
            embedding_matrix_path = os.path.join(os.path.dirname(padded_path), self.config.embedded_matrix_file)
            np.save(embedding_matrix_path, embedding_matrix)

            logging.info("Embedding matrix + tokenizer + padded sequences saved.")

            return EmbeddingArtifacts(
                padded_sequences_path=padded_path,
                tokenizer_path=tokenizer_path,
                embedding_matrix_path=embedding_matrix_path
            )

        except Exception as e:
            raise CustomException(e, sys)
