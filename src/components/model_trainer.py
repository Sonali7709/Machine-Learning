import os
import sys
from dataclasses import dataclass
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, f1_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Splitting train and test data into input and target")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(probability=True),
                "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                "CatBoost": CatBoostClassifier(verbose=0)
            }
            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            logging.info(f"Model report: {model_report}")
            best_model_name = max(model_report, key=lambda k: model_report[k]["Accuracy"])
            best_model_score = model_report[best_model_name]["Accuracy"]
        
            best_model = models[best_model_name]
            
            print(f"\n✅ Best Model: {best_model_name}")
            print(f"✅ Accuracy: {best_model_score:.4f}")
            print(f"✅ F1 Score: {model_report[best_model_name]['F1 Score']:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with accuracy above 60%")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted, average="weighted")

            return acc, f1
            
        except Exception as e:
            raise CustomException(e,sys)