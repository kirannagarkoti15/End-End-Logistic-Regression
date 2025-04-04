import pandas as pd
import joblib
from datetime import datetime
import os
from src.load_configuration import configuration

class PredictionOnNewData:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.curr_year = datetime.now().year
        self.model_features = self.config['model']['features']
        self.model_path = os.path.join(self.loc, "saved_models", "final_model_lr.joblib")
        self.scaler_path = os.path.join(self.loc, "saved_models", "scaler.joblib")
        self.prediction = os.path.join(self.loc, "output", "prediction.csv")


    def read_new_df(self):
        data = pd.read_csv(os.path.join(self.loc, "raw", "loan_data_for_prediction.csv"))
        return data
    
    def load_saved_models(self):
        model = joblib.load(self.model_path )
        scaler = joblib.load(self.scaler_path)
        return model, scaler
    
    def feature_creation(self, data):
        data = pd.get_dummies(data, columns=['person_gender', 'previous_loan_defaults_on_file', 'person_home_ownership', 'loan_intent'], drop_first=True)
        education_mapping = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
        data['person_education'] = data['person_education'].map(education_mapping)
        data = data.astype(float)
        for col in self.model_features :
            if col not in data.columns:
                data[col] = 0 
        return data
    
    def get_prediction(self):
        data_new = self.read_new_df()
        data = data_new.copy()
        model, scaler = self.load_saved_models()
        data = self.feature_creation(data)
        data = data[self.model_features]
        data_scaled = scaler.transform(data)
        loan_approval_prediction = model.predict(data_scaled)
        data_new["loan_status"] = loan_approval_prediction
        data_new.to_csv(self.prediction)
        return data_new