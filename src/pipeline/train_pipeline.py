import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import  ModelTrainerConfig, ModelTrainer
from src.utils import save_object



DATA_PATH = 'notebook/data/loan_data.csv'
TARGET_VARIABLE = 'Approval'




# test
if __name__ == "__main__":

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # here is updated
    modeltrainer = ModelTrainer()
    report, trained_models = modeltrainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    save_object(
        file_path=os.path.join('artifacts','model.pkl'),
        obj=trained_models
    ) 

    print(report)

