import os
import sys
import numpy as np
import pandas as pd


from src.logger import logging   
from src.exception import CustomException 

import dill

from sklearn.metrics import accuracy_score, f1_score

            
def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average="weighted")

            # ✅ Always print to terminal
            print(f"{model_name} → Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, F1: {f1:.4f}")

            report[model_name] = {
                "Accuracy": test_acc,
                "Train Score": train_acc,
                "F1 Score": f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
