import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact',"train_csv")
    test_data_path:str=os.path.join('artifact',"test_csv")
    raw_data_path:str=os.path.join('artifact',"data_csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        
        try:
            df = pd.read_csv("notebook/data/predictive_maintenance.csv")
            logging.info("read dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            
            )
            
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    # 1️⃣ Run data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2️⃣ Run data transformation
    transformation = DataTransformation()
    train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    # 3️⃣ Train models and get scores
    trainer = ModelTrainer()
    acc, f1 = trainer.initiate_model_trainer(train_array, test_array, preprocessor_path)

    # 4️⃣ Print scores to terminal
    print(f"\n✅ Final Best Model Accuracy: {acc:.4f}")
    print(f"✅ Final Best Model F1 Score: {f1:.4f}")