from hate_speech_classifier.config.configuration import ConfigurationManager
from hate_speech_classifier.components.stage_00_data_ingestion import DataIngestion
# from hate_speech_classifier.components.stage_01_data_transformation import DataTransformation
# from hate_speech_classifier.components.stage_02_model_trainer import ModelTrainer

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

    # Step 2: Data Transformation (Placeholder)
    # print("Starting: Data Transformation")
    # transform_config = config.get_data_transformation_config()
    # transformation = DataTransformation(transform_config, ingestion_artifacts)
    # transformed_artifacts = transformation.initiate()
    # print(f"Transformation Done! Output: {transformed_artifacts}")

    # Step 3: Model Training (Placeholder)
    # print("Starting: Model Training")
    # model_config = config.get_model_trainer_config()
    # trainer = ModelTrainer(model_config, transformed_artifacts)
    # model_artifacts = trainer.initiate()
    # print(f"Model Training Done! Model Path: {model_artifacts.model_path}")

if __name__ == "__main__":
    run_pipeline()