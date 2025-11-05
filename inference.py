import pandas as pd
import pickle

class FraudDetector:
    def __init__(self, model_path='models/fraud_model.pkl', scaler_path='models/scaler.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, data):
        X = data.copy()
        expected_cols = list(self.scaler.feature_names_in_)
        
        rename_map = {col.lower(): col for col in expected_cols}
        X.rename(columns=rename_map, inplace=True)
        
        X[expected_cols] = self.scaler.transform(X[expected_cols])
        return self.model.predict(X)

    def predict_batch(self, transactions):
        X = transactions.copy()
        expected_cols = list(self.scaler.feature_names_in_)
        
        rename_map = {col.lower(): col for col in expected_cols}
        X.rename(columns=rename_map, inplace=True)
        
        X_scaled = X[expected_cols].copy()
        X_scaled[expected_cols] = self.scaler.transform(X_scaled[expected_cols])
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'is_fraud': bool(predictions[i]),
                'fraud_probability': probabilities[i][1]
            })
        
        return results