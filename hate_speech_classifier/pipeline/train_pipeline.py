from hate_speech_classifier.config.configuration import ConfigurationManager
from hate_speech_classifier.components.stage_00_data_ingestion import DataIngestion
from hate_speech_classifier.components.stage_01_preprocessing import Preprocessing

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
    cleaned_file_path = preprocessor.initiate()
    print(f"Preprocessing Done! Cleaned File: {cleaned_file_path}")


    
if __name__ == "__main__":
    run_pipeline()