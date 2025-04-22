from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionConfig:
    bucket_name: str
    zip_file_name: str
    artifacts_dir: str
    ingestion_dir: str
    imbalance_file_name: str
    raw_file_name: str

@dataclass
class DataTransformationConfig:
    drop_columns: List[str]
    label_column: str
    text_column: str
    transformed_file_name: str

@dataclass
class TokenizerConfig:
    max_num_words: int
    max_sequence_length: int

@dataclass
class EmbeddingConfig:
    glove_path: str
    embedding_dim: int
    trainable: bool

@dataclass
class ModelTrainerConfig:
    conv1d_filters: int
    kernel_size: int
    pool_size: int
    lstm_units: int
    dropout_rate: float
    dense_units: int
    l2_regularization: float
    final_activation: str
    batch_size: int
    epochs: int
    loss: str
    metrics: list
    validation_split: float
    random_state: int
    model_name: str

@dataclass
class EvaluationConfig:
    threshold: float



# from dataclasses import dataclass
# from hate_speech_classifier.constants.global_constants import *
# import os 


# @dataclass
# class DataIngestionConfig:
#     def __init__(self):
#         self.BUCKET_NAME: str = BUCKET_NAME
#         self.ZIP_FILE_NAME: str = ZIP_FILE_NAME
#         self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
#         self.IMBALANCED_DATA_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
#         self.RAW_DATA_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_RAW_DATA_DIR)
#         self.ZIP_FILE_DIR: str = self.DATA_INGESTION_ARTIFACTS_DIR
#         self.ZIP_FILE_PATH: str = os.path.join(self.ZIP_FILE_DIR, self.ZIP_FILE_NAME)

# @dataclass
# class DataTransformationConfig:
#     def __init__(self):
#         self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
#         self.TRANSFORMED_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
#         self.AXIS = AXIS
#         self.INPLACE = INPLACE
#         self.DROP_COLUMNS = DROP_COLUMNS
#         self.CLASS = CLASS
#         self.LABEL = LABEL
#         self.TEXT = TEXT

# @dataclass
# class ModelTrainerConfig:
#     def __init__(self):
#         self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
#         self.TRAINED_MODEL_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
#         self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
#         self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
#         self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)

#         # Architecture + Hyperparameters
#         self.MAX_NUM_WORDS = MAX_NUM_WORDS
#         self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        
#         self.LOSS = LOSS
#         self.METRICS = METRICS
#         self.ACTIVATION = FINAL_ACTIVATION
#         self.RANDOM_STATE = RANDOM_STATE
#         self.EPOCHS = EPOCHS
#         self.BATCH_SIZE = BATCH_SIZE
#         self.VALIDATION_SPLIT = VALIDATION_SPLIT

#         # Embedding
#         self.GLOVE_PATH = GLOVE_PATH
#         self.EMBEDDING_DIM = EMBEDDING_DIM
#         self.EMBEDDING_TRAINABLE = EMBEDDING_TRAINABLE

#         # Architecture Constants
#         self.CONV1D_FILTERS = CONV1D_FILTERS
#         self.KERNEL_SIZE = KERNEL_SIZE
#         self.POOL_SIZE = POOL_SIZE
#         self.LSTM_UNITS = LSTM_UNITS
#         self.DROPOUT_RATE = DROPOUT_RATE
#         self.DENSE_UNITS = DENSE_UNITS
#         self.L2_REGULARIZATION = L2_REGULARIZATION

# @dataclass
# class ModelEvaluationConfig:
#     def __init__(self):
#         self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
#         self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR, BEST_MODEL_DIR)
#         self.MODEL_EVALUATION_FILE_NAME: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR, MODEL_EVALUATION_FILE_NAME)
#         self.THRESHOLD = THRESHOLD
#         self.BUCKET_NAME = BUCKET_NAME
#         self.MODEL_NAME = MODEL_NAME

# @dataclass
# class ModelPusherConfig:
#     def __init__(self):
#         self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_NAME)
#         self.BUCKET_NAME = BUCKET_NAME
#         self.MODEL_NAME = MODEL_NAME