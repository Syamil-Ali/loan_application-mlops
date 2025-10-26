import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
import pandas as pd



def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:

        report = []
        trained_models = {}

        for name, model in models.items():

            print(f"Training {name} ...")
            trained_models[name] = model
        
            model.fit(X_train, y_train)


            # ---- Test Set ----
            y_pred_test = model.predict(X_test)
            y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc_test = accuracy_score(y_test, y_pred_test)
            prec_test = precision_score(y_test, y_pred_test, zero_division=0)
            rec_test = recall_score(y_test, y_pred_test, zero_division=0)
            f1_test = f1_score(y_test, y_pred_test, zero_division=0)
            roc_test = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else None

            # Append results
            report.append({
                "Model": name,
                "Test Accuracy": acc_test,
                "Test Precision": prec_test,
                "Test Recall": rec_test,
                "Test F1": f1_test,
                "Test ROC_AUC": roc_test,
            })

        return report, trained_models 

    except Exception as e:
        raise CustomException(e, sys)


# ------------------------------------ #

@dataclass
class ModelTrainerConfig:

    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

        

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            
            logging.info("Split training and test input data")
            #X_train, y_train, X_test, y_test = (
            #    train_array[:,:-1], # take all column except last
            #    train_array[:,-1], # take last column
            #    test_array[:,:-1],
            #    test_array[:,-1]
            #)

            random_state = 42

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
                "SVM": SVC(probability=True, random_state=random_state),
                "KNN": KNeighborsClassifier()
            }

            report, trained_models = evaluate_models(X_train, y_train, X_test, y_test, models) 


            # right now im converted to df because Im lazy but next time, do the TODO
            # Convert to DataFrame
            results_df = pd.DataFrame(report)

            # sort the df to based on the highest validation precision
            results_df.sort_values(by=['Test F1'], ascending=False, inplace=True)

            # export
            #results_df.to_csv(f'{self.model_trainer_config.trained_model_file_path}/result.csv', index=False)
            results_df.to_csv("artifacts/result.csv",
                index=False
            )


            ### TODO: TO SORT THE BEST MODEL SCORE FROM THE DICT

            # 1. To sort the report to based on the highest validation precision
            #report.sort_values(by=['Validation Precision'], ascending=False, inplace=True)
            # 
            # 2. To get the best model   
            # 
            # 3 Show exception if all model accuracy is lower than 60%

            # if best model < 0.6:
            # raise CustomException("No Best Model Found")  
            # 

            logging.info("Best found model on both training and testing dataset") 

            # save the model
            #save_object(
            #    file_path=self.model_trainer_config.trained_model_file_path
            #    obj = best model
            #)

            return report, trained_models

            
        except Exception as e:
            raise CustomException(e, sys)
            

    
