import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os
import joblib

from src.utils import save_object

max_k_cluster = 50

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:


    def __init__(self):
        self.config = DataTransformationConfig()

    def text_column_transformer_train(self, X: pd.DataFrame, column_name: str):
        logging.info("Start clustering Text column")

        texts = X[column_name].dropna().astype(str)
        vectorizer = TfidfVectorizer(stop_words="english")
        text_vectorize = vectorizer.fit_transform(texts)

        # Try k values
        K = range(1, max_k_cluster)
        inertias = []
        for k in K:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(text_vectorize)
            inertias.append(model.inertia_)

        # Find elbow
        kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
        best_k = kl.elbow if kl.elbow else 5
        logging.info(f"Optimal clusters: {best_k}")

        knn_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        knn_model.fit(text_vectorize)

        return vectorizer, knn_model

    def fit_scaler(self, X: pd.DataFrame, numeric_columns: list):
        scaler = StandardScaler()
        scaler.fit(X[numeric_columns])
        return scaler

    def scaler_text_transformation(self, X: pd.DataFrame, text_col: str, numeric_columns: list):
        vectorizer, text_knn_model = self.text_column_transformer_train(X, text_col)
        scaler = self.fit_scaler(X, numeric_columns)
        return vectorizer, text_knn_model, scaler


    def get_data_transformer_object(self, X: pd.DataFrame, text_column:str, numerical_columns:list, vectorizer,text_knn_model, scaler):
        try:

            X[text_column] = text_knn_model.predict(vectorizer.transform(X[text_column]))

            # Transform categorical (better: OneHotEncoder)
            X['Employment_Status'] = X['Employment_Status'].replace(
                    {'unemployed': 0, 'employed': 1}
                )

            # Scale numerics
            X[numerical_columns] = scaler.transform(X[numerical_columns])
        

            return X

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_columns = ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio']
            categorical_columns = ['Employment_Status']
            text_column = 'Text'
            target_col = 'Approval'


            # Fit transformers on train
            logging.info("Obtaining preprocessing object")
            print('preprocessing object')
            vectorizer, text_knn_model, scaler = self.scaler_text_transformation(train_df, text_column, numerical_columns)
            print('done object')


            # split the X and y
            X_train = train_df.drop(columns=[target_col]).copy()
            y_train = train_df[target_col].copy()

            X_test = test_df.drop(columns=[target_col]).copy()
            y_test = test_df[target_col].copy()


            # Transformer for both train and test
            X_train = self.get_data_transformer_object(X_train, text_column, numerical_columns, vectorizer,text_knn_model, scaler)
            X_test = self.get_data_transformer_object(X_test, text_column, numerical_columns, vectorizer,text_knn_model, scaler)

            # Ensure folder exists
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            preprocessor = {
                "vectorizer": vectorizer,
                "text_knn_model": text_knn_model,
                "scaler": scaler
            }

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            ) 

            logging.info(f"Preprocessor saved to {self.config.preprocessor_obj_file_path}")

            return(
                X_train,y_train, 
                X_test, y_test,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
