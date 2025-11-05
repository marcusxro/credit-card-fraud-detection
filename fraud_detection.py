"""
Credit Card Fraud Detection Model Training
This script trains multiple machine learning models to detect fraudulent transactions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    A class to handle Credit Card Fraud Detection model training and evaluation.
    """
    
    def __init__(self, data_path='creditcard.csv'):
        """
        Initialize the fraud detection model.
        
        Args:
            data_path (str): Path to the credit card dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and perform initial exploration of the dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        print("\n\nDataset Info:")
        print(self.data.info())
        
        print("\n\nClass Distribution:")
        print(self.data['Class'].value_counts())
        print(f"\nPercentage of Fraudulent Transactions: {(self.data['Class'].sum() / len(self.data)) * 100:.4f}%")
        
        # Check for missing values
        print("\n\nMissing Values:")
        print(self.data.isnull().sum())
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations for exploratory data analysis."""
        print("\nGenerating visualizations...")
        
        # Create output directory for plots
        os.makedirs('plots', exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(8, 6))
        self.data['Class'].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.title('Class Distribution (0: Normal, 1: Fraud)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/class_distribution.png')
        print("✓ Saved: plots/class_distribution.png")
        plt.close()
        
        # Amount distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.data[self.data['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='Normal')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.title('Normal Transactions Amount Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(self.data[self.data['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.title('Fraudulent Transactions Amount Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/amount_distribution.png')
        print("✓ Saved: plots/amount_distribution.png")
        plt.close()
        
        # Time distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.data[self.data['Class'] == 0]['Time'], bins=50, alpha=0.7, label='Normal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Normal Transactions Time Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(self.data[self.data['Class'] == 1]['Time'], bins=50, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Fraudulent Transactions Time Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/time_distribution.png')
        print("✓ Saved: plots/time_distribution.png")
        plt.close()
        
        # Correlation heatmap (sample for visibility)
        plt.figure(figsize=(12, 10))
        correlation = self.data.corr()
        sns.heatmap(correlation, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png')
        print("✓ Saved: plots/correlation_heatmap.png")
        plt.close()
        
    def preprocess_data(self, test_size=0.3, use_smote=True, random_state=42):
        """
        Preprocess the data: scaling, train-test split, and handling imbalance.
        
        Args:
            test_size (float): Proportion of dataset for testing
            use_smote (bool): Whether to use SMOTE for handling class imbalance
            random_state (int): Random state for reproducibility
        """
        print("\n" + "="*50)
        print("PREPROCESSING DATA")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"\nTraining set fraud cases: {self.y_train.sum()}")
        print(f"Testing set fraud cases: {self.y_test.sum()}")
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("\nApplying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"After SMOTE - Training set size: {self.X_train.shape[0]}")
            print(f"After SMOTE - Fraud cases: {self.y_train.sum()}")
            print(f"After SMOTE - Normal cases: {len(self.y_train) - self.y_train.sum()}")
        
        print("\n✓ Preprocessing completed successfully!")
        
    def train_models(self):
        """Train multiple machine learning models."""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("✓ Logistic Regression training completed")
        
        # Random Forest
        print("\n2. Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✓ Random Forest training completed")
        
        print("\n✓ All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all trained models and generate performance metrics."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}")
            print(f"{'='*50}")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print metrics
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            print("\n\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            print(f"\nTrue Negatives: {cm[0][0]}")
            print(f"False Positives: {cm[0][1]}")
            print(f"False Negatives: {cm[1][0]}")
            print(f"True Positives: {cm[1][1]}")
            
    def plot_model_comparison(self):
        """Create comparison plots for all models."""
        print("\n\nGenerating model comparison plots...")
        
        os.makedirs('plots', exist_ok=True)
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            axes[idx].bar(model_names, values, color=['blue', 'green'])
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        print("✓ Saved: plots/model_comparison.png")
        plt.close()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for model_name in model_names:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = self.results[model_name]['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/roc_curves.png')
        print("✓ Saved: plots/roc_curves.png")
        plt.close()
        
        # Confusion matrices
        fig, axes = plt.subplots(1, len(model_names), figsize=(15, 5))
        if len(model_names) == 1:
            axes = [axes]
            
        for idx, model_name in enumerate(model_names):
            y_pred = self.results[model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png')
        print("✓ Saved: plots/confusion_matrices.png")
        plt.close()
        
    def save_models(self):
        """Save trained models and scaler to disk."""
        print("\n\nSaving models...")
        os.makedirs('saved_models', exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"saved_models/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"✓ Saved: {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, 'saved_models/scaler.pkl')
        print("✓ Saved: saved_models/scaler.pkl")
        
    def run_full_pipeline(self, use_smote=True):
        """
        Run the complete fraud detection pipeline.
        
        Args:
            use_smote (bool): Whether to use SMOTE for handling class imbalance
        """
        print("\n" + "="*70)
        print("CREDIT CARD FRAUD DETECTION - FULL PIPELINE")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Visualize data
        self.visualize_data()
        
        # Preprocess
        self.preprocess_data(use_smote=use_smote)
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Plot comparisons
        self.plot_model_comparison()
        
        # Save models
        self.save_models()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nResults saved in:")
        print("  - plots/ directory: visualizations and model comparisons")
        print("  - saved_models/ directory: trained models and scaler")


def main():
    """Main function to run the fraud detection pipeline."""
    import sys
    
    # Check if dataset path is provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'creditcard.csv'
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found!")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\nAnd place it in the project directory, or provide the path as an argument:")
        print(f"python {sys.argv[0]} /path/to/creditcard.csv")
        sys.exit(1)
    
    # Create and run the model
    fraud_detector = FraudDetectionModel(data_path=data_path)
    fraud_detector.run_full_pipeline(use_smote=True)


if __name__ == "__main__":
    main()
