"""
Credit Card Fraud Detection - Prediction Script
This script loads a trained model and makes predictions on new transactions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys


class FraudPredictor:
    """
    A class to load trained models and make fraud predictions on new transactions.
    """
    
    def __init__(self, model_path='saved_models/random_forest_model.pkl', 
                 scaler_path='saved_models/scaler.pkl'):
        """
        Initialize the fraud predictor.
        
        Args:
            model_path (str): Path to the trained model file
            scaler_path (str): Path to the scaler file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def load_model(self):
        """Load the trained model and scaler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        print("Loading model and scaler...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        print("✓ Model and scaler loaded successfully!")
        
    def predict_single(self, transaction_features):
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction_features (array-like): Features of the transaction
            
        Returns:
            dict: Prediction results including class and probability
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure features are in the right shape
        features = np.array(transaction_features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability[1]),
            'normal_probability': float(probability[0]),
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud for multiple transactions from a DataFrame.
        
        Args:
            transactions_df (pd.DataFrame): DataFrame containing transaction features
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\nMaking predictions for {len(transactions_df)} transactions...")
        
        # Remove 'Class' column if it exists (for testing with labeled data)
        has_labels = 'Class' in transactions_df.columns
        if has_labels:
            y_true = transactions_df['Class']
            X = transactions_df.drop('Class', axis=1)
        else:
            X = transactions_df.copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results = transactions_df.copy()
        results['Predicted_Class'] = predictions
        results['Fraud_Probability'] = probabilities
        results['Risk_Level'] = [self._get_risk_level(p) for p in probabilities]
        
        # Print summary
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total transactions: {len(results)}")
        print(f"Predicted as fraud: {predictions.sum()} ({predictions.sum()/len(results)*100:.2f}%)")
        print(f"Predicted as normal: {len(results) - predictions.sum()} ({(len(results) - predictions.sum())/len(results)*100:.2f}%)")
        
        # Risk level breakdown
        print("\nRisk Level Distribution:")
        print(results['Risk_Level'].value_counts().to_string())
        
        if has_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions)
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            
            print("\n" + "="*50)
            print("PERFORMANCE METRICS (with true labels)")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
        
        return results
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on fraud probability.
        
        Args:
            probability (float): Fraud probability
            
        Returns:
            str: Risk level
        """
        if probability < 0.3:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        elif probability < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def display_prediction(self, result):
        """
        Display prediction result in a formatted way.
        
        Args:
            result (dict): Prediction result from predict_single()
        """
        print("\n" + "="*50)
        print("FRAUD DETECTION RESULT")
        print("="*50)
        print(f"\nPrediction: {'🚨 FRAUD DETECTED' if result['is_fraud'] else '✓ Normal Transaction'}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
        print(f"Normal Probability: {result['normal_probability']:.4f} ({result['normal_probability']*100:.2f}%)")
        print(f"Risk Level: {result['risk_level']}")
        
        if result['risk_level'] in ['High', 'Critical']:
            print("\n⚠️  WARNING: This transaction requires immediate attention!")
        elif result['risk_level'] == 'Medium':
            print("\n⚠️  CAUTION: This transaction may need further investigation.")
        else:
            print("\n✓ This transaction appears to be legitimate.")


def main():
    """Main function for making predictions."""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PREDICTION")
    print("="*70)
    
    # Check for required files
    if not os.path.exists('saved_models'):
        print("\nError: 'saved_models' directory not found!")
        print("Please train the model first by running: python fraud_detection.py")
        sys.exit(1)
    
    # Initialize predictor
    predictor = FraudPredictor()
    
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the model first by running: python fraud_detection.py")
        sys.exit(1)
    
    # Example usage
    if len(sys.argv) > 1:
        # Batch prediction from CSV file
        csv_path = sys.argv[1]
        
        if not os.path.exists(csv_path):
            print(f"\nError: File '{csv_path}' not found!")
            sys.exit(1)
        
        print(f"\nLoading transactions from: {csv_path}")
        transactions = pd.read_csv(csv_path)
        
        # Make predictions
        results = predictor.predict_batch(transactions)
        
        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")
        
        # Display high-risk transactions
        high_risk = results[results['Risk_Level'].isin(['High', 'Critical'])]
        if len(high_risk) > 0:
            print("\n" + "="*50)
            print("HIGH-RISK TRANSACTIONS")
            print("="*50)
            print(high_risk[['Time', 'Amount', 'Fraud_Probability', 'Risk_Level']].head(10))
    
    else:
        # Interactive mode for single prediction
        print("\n" + "="*50)
        print("INTERACTIVE PREDICTION MODE")
        print("="*50)
        print("\nNote: This requires a transaction with 30 features (Time, V1-V28, Amount)")
        print("For batch predictions, run: python predict.py <path_to_transactions.csv>")
        
        # Example with dummy data
        print("\n\nExample prediction with sample data:")
        # Create a sample transaction (30 features)
        sample_features = np.random.randn(30)
        sample_features[0] = 100  # Time
        sample_features[-1] = 50.0  # Amount
        
        result = predictor.predict_single(sample_features)
        predictor.display_prediction(result)


if __name__ == "__main__":
    main()
