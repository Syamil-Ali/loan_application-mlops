import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        
        # Do the prediction here

        # 1. load the preprocessor here
        preprocessor = load_object('artifacts/preprocessor.pkl')
        
        # preprocessor = {
        #        "vectorizer": vectorizer,
        #        "text_knn_model": text_knn_model,
        #        "scaler": scaler
        #    }

        # 2. load model
        model = load_object('artifacts/model.pkl')

        # 3. transform the input
        numerical_columns = ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio']

        features['Text'] = preprocessor['text_knn_model'].predict(preprocessor['vectorizer'].transform(features['Text']))
        features['Employment_Status'] = features['Employment_Status'].replace(
                    {'unemployed': 0, 'employed': 1}
                )
        features[numerical_columns] = preprocessor['scaler'].transform(features[numerical_columns])

        result = model['Logistic Regression'].predict(features)

        return result

class CustomData:

    def __init__(self,
        text: str,
        income: int,
        credit_score: int,
        loan_amount: int,
        dti_ratio: float,
        employment_status: str):

        self.text = text
        self.income = income
        self.credit_score = credit_score
        self.loan_amount = loan_amount
        self.dti_ratio = dti_ratio
        self.employment_status = employment_status

    
    def get_data_as_dataframe(self):

        # convert to df
        try:
            custom_data_input_dict = {
                "Text": [self.text],
                "Income": [self.income],
                "Credit_Score": [self.credit_score],
                "Loan_Amount": [self.loan_amount],
                "DTI_Ratio": [self.dti_ratio],
                "Employment_Status": [self.employment_status]
            }
            #Text,Income,Credit_Score,Loan_Amount,DTI_Ratio,Employment_Status,Approval

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":

    # 1. Start with Custom Data
    custom_data = CustomData(
        'I need a loan to pay for an international vacation with my family.',26556,581,8314,79.26,'employed'
    )
    
    result_df = custom_data.get_data_as_dataframe()

    # 2. Pass the result to Predict
    predict_pipeline = PredictPipeline()

    result_predict = predict_pipeline.predict(result_df)


    print('Result DF:')
    print(result_df)
    print('----------')
    print('Result Prediction:')
    print(result_predict)
