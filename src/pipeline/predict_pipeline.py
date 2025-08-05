import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            
            model_path = 'artifact\\model.pkl'
            preprocessor_path = 'artifact\\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self, 
                Type: str,
                Air_temperature_K: float,
                Process_temperature_K: float,
                Rotational_speed_rpm: int,
                Torque_Nm: float,
                Tool_wear_min: int,
                Target: float):
        
        self.Type = Type
        self.Air_temperature_K = Air_temperature_K
        self.Process_temperature_K = Process_temperature_K
        self.Rotational_speed_rpm = Rotational_speed_rpm
        self.Torque_Nm = Torque_Nm
        self.Tool_wear_min = Tool_wear_min
        self.Target = Target

    def get_data_as_data_frame(self):
        try:
            # Create DataFrame from input features
            custom_data_input_dict = {
                "Type": [self.Type],
                "Air temperature [K]": [self.Air_temperature_K],
                "Process temperature [K]": [self.Process_temperature_K],
                "Rotational speed [rpm]": [self.Rotational_speed_rpm],
                "Torque [Nm]": [self.Torque_Nm],
                "Tool wear [min]": [self.Tool_wear_min],
                "Target": [self.Target]
            }
            
            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
