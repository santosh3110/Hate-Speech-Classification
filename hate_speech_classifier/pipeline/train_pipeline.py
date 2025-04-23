from hate_speech_classifier.config.configuration import ConfigurationManager
from hate_speech_classifier.components.stage_00_data_ingestion import DataIngestion
from hate_speech_classifier.components.stage_01_preprocessing import Preprocessing
from hate_speech_classifier.components.stage_02_embeddings import EmbeddingLayer
from hate_speech_classifier.components.stage_03_model_building import ModelBuilder
from hate_speech_classifier.components.stage_04_model_training import ModelTrainer
from hate_speech_classifier.components.stage_05_model_evaluation import ModelEvaluation
from hate_speech_classifier.components.stage_06_model_pusher import ModelPusher

from hate_speech_classifier.entity.artifact_entity import PreprocessingArtifacts,EmbeddingArtifacts,ModelEvaluationArtifacts


def run_pipeline():
    config = ConfigurationManager()

    # Step 1: Data Ingestion
    print(" Starting: Data Ingestion")
    print(" Pipeline started...")
    data_ingestion_config = config.get_data_ingestion_config()
    print(" Config loaded for Data Ingestion")

    ingestion = DataIngestion(data_ingestion_config)
    ingestion_artifacts = ingestion.initiate()
    print(f" Data Ingestion Done! Files: {ingestion_artifacts}")

    # STEP 2: Preprocessing
    print("Starting: Preprocessing")
    preprocess_config = config.get_preprocessing_config()
    preprocessor = Preprocessing(preprocess_config, ingestion_artifacts)
    preprocessing_artifacts = preprocessor.initiate()
    print(f"Preprocessing Done! Cleaned File: {preprocessing_artifacts.cleaned_data_path}")

    # STEP 3: Embeddings
    print("Starting: Embedding Setup")
    embedding_config = config.get_embedding_config()
    embedder = EmbeddingLayer(embedding_config, preprocessing_artifacts)
    embedding_artifacts = embedder.initiate()
    print(f"Embedding Setup Done!\nTokenizer: {embedding_artifacts.tokenizer_path}")

    # Step 4: Build Model
    print("Starting: Model Building")
    model_config = config.get_model_config()
    builder = ModelBuilder(model_config, embedding_artifacts)
    model = builder.build_model()
    print("Model Built Successfully")
  
    # Step 5: Model Training
    print("Starting: Model Training")
    model_config = config.get_model_training_config()
    trainer = ModelTrainer(model_config, embedding_artifacts)
    training_artifacts = trainer.initiate()
    print(f"Model Training Done! Saved to: {training_artifacts.trained_model_path}")

    # Step 6: Model Evaluation
    print("Starting: Model Evaluation")
    eval_config = config.get_model_evaluation_config()
    class_threshold = model_config.class_threshold  # from training config
    evaluator = ModelEvaluation(eval_config, training_artifacts, embedding_artifacts, class_threshold)
    is_model_accepted = evaluator.initiate()

    if is_model_accepted:
        print("Model accepted. Proceeding to push model...")
        # Step 7: Push Model to GCS
        pusher_config = config.get_model_pusher_config()
        pusher = ModelPusher(pusher_config, training_artifacts)
        pusher.initiate()
        print("Model pushed to GCS.")
    else:
        print("Model rejected. Not pushing to GCS.")


    






    
if __name__ == "__main__":
    run_pipeline()