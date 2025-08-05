import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """this is responsible for data transformation"""
        try:
            numerical_columns=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Target']
            categorical_columns=['Type',]
            
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ]            
            )
            
            logging.info("Numerical column standard scaling completed.")
            
            logging.info("Categorical columns encoding completed.")
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
       
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            drop_columns = ['UDI', 'Product ID']
            logging.info(f"Dropping columns: {drop_columns}")

            train_df.drop(columns=drop_columns, errors='ignore', inplace=True)
            test_df.drop(columns=drop_columns, errors='ignore', inplace=True)

            temp_columns = ['Air temperature [K]', 'Process temperature [K]']
            for col in temp_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col] - 273.15
                if col in test_df.columns:
                    test_df[col] = test_df[col] - 273.15

            logging.info("Converted temperature columns from Kelvin to Celsius.")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Failure Type"
            label_encoder=LabelEncoder()
            
            train_df[target_column_name] = label_encoder.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = label_encoder.transform(test_df[target_column_name])
            logging.info(f"Encoded classes: {list(label_encoder.classes_)}")
            logging.info(f"First 10 encoded target values (train): {train_df[target_column_name].head(10).tolist()}")
            logging.info(f"First 10 encoded target values (test): {test_df[target_column_name].head(10).tolist()}")
            # Preprocessor
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine processed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info(f"Saving preprocessing object to: {os.path.abspath(self.data_transformation_config.preprocessor_obj_file_path)}")

        
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("✅ Preprocessing object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


