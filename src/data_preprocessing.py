import os
import pandas as pd
from datetime import datetime
from src.load_configuration import configuration


class DataPrepration:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.curr_year = datetime.now().year
        self.numerical_features = self.config['data']['numerical_features']


    def data_read(self):
        data = pd.read_csv(os.path.join(self.loc, "raw", "loan_data.csv"))
        return data

    def feature_creation(self, data):
        data = pd.get_dummies(data, columns=['person_gender', 'previous_loan_defaults_on_file', 'person_home_ownership', 'loan_intent'], drop_first=True)
        education_mapping = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
        data['person_education'] = data['person_education'].map(education_mapping)
        data = data.astype(float)
        return data

    def outlier_removal(self, data):
        # Removing outliers for both classes seperately
        class_0 = data[data['loan_status'] == 0]
        class_1 = data[data['loan_status'] == 1]

        Q1_0 = class_0[self.numerical_features].quantile(0.25)
        Q3_0 = class_0[self.numerical_features].quantile(0.75)
        IQR_0 = Q3_0 - Q1_0

        Q1_1 = class_1[self.numerical_features].quantile(0.25)
        Q3_1 = class_1[self.numerical_features].quantile(0.75)
        IQR_1 = Q3_1 - Q1_1

        lower_bound_0 = Q1_0 - 1.5 * IQR_0
        upper_bound_0 = Q3_0 + 1.5 * IQR_0

        lower_bound_1 = Q1_1 - 1.5 * IQR_1
        upper_bound_1 = Q3_1 + 1.5 * IQR_1

        class_0_filtered = class_0[~((class_0[self.numerical_features] < lower_bound_0) | (class_0[self.numerical_features] > upper_bound_0)).any(axis=1)]
        class_1_filtered = class_1[~((class_1[self.numerical_features] < lower_bound_1) | (class_1[self.numerical_features] > upper_bound_1)).any(axis=1)]

        data = pd.concat([class_0_filtered, class_1_filtered])
        return data
    
    def data_process(self):
        data = self.data_read()
        data = self.feature_creation(data)
        data = self.outlier_removal(data)
        data.drop(columns=['person_age', 'loan_percent_income'], inplace=True)
        data.to_csv(self.loc + '/processed_data/processed_data.csv', index=False)
        return data
