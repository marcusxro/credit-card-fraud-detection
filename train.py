import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

def load_and_prepare_data(filepath):
    print("loading")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
    print(f"Legitimate cases: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")
    
    return df

def preprocess_data(df):
    print("\nPreprocessing data...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    
    return X, y, scaler

def train_model(X_train, y_train):
    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

def save_model(model, scaler, output_dir='models'):
    """Save trained model and scaler"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {output_dir}/fraud_model.pkl")
    print(f"Scaler saved to {output_dir}/scaler.pkl")

def main():
    DATA_PATH = 'data/creditcard.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    df = load_and_prepare_data(DATA_PATH)
    
    X, y, scaler = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)
    
    save_model(model, scaler)

if __name__ == "__main__":
    main()