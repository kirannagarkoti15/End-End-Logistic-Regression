import os
import pandas as pd
import numpy as np
from src.load_configuration import configuration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report, roc_auc_score
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore") 


class ModelBuild:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.target = self.config['data']['target']
        self.model_path = os.path.join(self.loc, "saved_models", "final_model_lr.joblib")
        self.scaler_path = os.path.join(self.loc, "saved_models", "scaler.joblib")


    def read_processed_df(self):
        data = pd.read_csv(os.path.join(self.loc, "processed_data", "processed_data.csv"))
        return data
    
    def REF(self, data):
        X = data.drop(columns=[self.target])
        y = data[self.target]
        model = LogisticRegression(solver='liblinear')
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(model, n_features_to_select=5))])
        pipeline.fit(X, y)
        rfe = pipeline.named_steps['feature_selection']
        selected_features = X.columns[rfe.support_]
        return X, y, selected_features
    
    def statndardization(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled
    
    def evaluate_model(self, y_true, y_pred, y_prob=None, dataset_type="Test"):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\n{dataset_type} Set Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        if y_prob is not None:
            auc_roc = roc_auc_score(y_true, y_prob)
            print(f"AUC-ROC: {auc_roc:.4f}")


    def save_models(self, model, scaler):
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
    
    def train_model(self):
        data = self.read_processed_df()
        X, y, selected_features = self.REF(data)
        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=14)
        scaler, X_train_scaled, X_test_scaled = self.statndardization(X_train, X_test)
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]  
        print("Train Classification Report:")
        print(classification_report(y_train, y_train_pred))
        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred))
        self.evaluate_model(y_train, y_train_pred, y_train_prob, dataset_type="Train")
        self.evaluate_model(y_test, y_test_pred, y_test_prob, dataset_type="Test")           
        self.save_models(model, scaler)
        return model