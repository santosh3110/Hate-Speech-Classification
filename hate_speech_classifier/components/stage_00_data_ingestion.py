import os
import sys
import shutil
from zipfile import ZipFile

from hate_speech_classifier.logger.log import logging
from hate_speech_classifier.exception.exception_handler import CustomException
from hate_speech_classifier.entity.config_entity import DataIngestionConfig
from hate_speech_classifier.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            logging.info("Downloading ZIP from GCS...")
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            command = f"gsutil cp gs://{self.config.bucket_name}/{self.config.zip_file_name} {self.config.artifacts_dir}/"
            os.system(command)
            logging.info("Download complete.")
        except Exception as e:
            raise CustomException(e, sys)

    def unzip_data(self):
        try:
            logging.info(" Unzipping dataset...")

            zip_path = os.path.join(self.config.artifacts_dir, self.config.zip_file_name)
            temp_extract_path = os.path.join(self.config.artifacts_dir, "extracted")
            os.makedirs(temp_extract_path, exist_ok=True)

            #  Unzip
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_path)

            # Destination folder for .csv files
            final_data_path = os.path.join(self.config.artifacts_dir, self.config.ingestion_dir)
            os.makedirs(final_data_path, exist_ok=True)

            # Find all .csv and move to ingestion dir
            found_csvs = []
            for root, dirs, files in os.walk(temp_extract_path):
                for file in files:
                    if file.endswith(".csv"):
                        source_file = os.path.join(root, file)
                        dest_file = os.path.join(final_data_path, file)
                        shutil.move(source_file, dest_file)
                        found_csvs.append(file)

            # Clean temp folder
            shutil.rmtree(temp_extract_path)

            if not found_csvs:
                raise FileNotFoundError("No .csv files found inside the extracted zip!")

            imbalance_path = os.path.join(final_data_path, self.config.imbalance_file_name)
            raw_path = os.path.join(final_data_path, self.config.raw_file_name)

            logging.info(f" Files moved to: {final_data_path}")
            logging.info(f"   - {imbalance_path}")
            logging.info(f"   - {raw_path}")

            return DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_path,
                raw_data_file_path=raw_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate(self) -> DataIngestionArtifacts:
        try:
            logging.info("Initiating full data ingestion stage...")

            self.download_data()
            artifacts = self.unzip_data()

            logging.info(f"Data Ingestion Success! Artifacts: {artifacts}")
            return artifacts

        except Exception as e:
            raise CustomException(e, sys)
