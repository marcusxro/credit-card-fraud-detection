"""
Data preprocessing utilities for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


class DataPreprocessor:
    """
    Utility class for preprocessing credit card transaction data.
    """
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method (str): Type of scaling ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler()
        
    def _get_scaler(self):
        """Get the appropriate scaler based on the method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaling_method, StandardScaler())
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using the specified scaling method.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Scaled training and testing features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X, y, method='smote', random_state=42):
        """
        Handle class imbalance using various sampling techniques.
        
        Args:
            X: Features
            y: Target variable
            method (str): Sampling method ('smote', 'adasyn', 'undersample', 'smotetomek')
            random_state (int): Random state for reproducibility
            
        Returns:
            Resampled X and y
        """
        methods = {
            'smote': SMOTE(random_state=random_state),
            'adasyn': ADASYN(random_state=random_state),
            'undersample': RandomUnderSampler(random_state=random_state),
            'smotetomek': SMOTETomek(random_state=random_state)
        }
        
        sampler = methods.get(method, SMOTE(random_state=random_state))
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"Original dataset shape: {X.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Original class distribution: {np.bincount(y)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def remove_outliers(self, df, columns, n_std=3):
        """
        Remove outliers from specified columns using z-score method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): List of column names to check for outliers
            n_std (int): Number of standard deviations for outlier threshold
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean = df_clean[
                (df_clean[col] >= mean - n_std * std) & 
                (df_clean[col] <= mean + n_std * std)
            ]
        
        print(f"Removed {len(df) - len(df_clean)} outliers")
        return df_clean
    
    def create_features(self, df):
        """
        Create additional features from existing ones.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            DataFrame with additional features
        """
        df_enhanced = df.copy()
        
        # Time-based features
        df_enhanced['Hour'] = (df_enhanced['Time'] / 3600) % 24
        df_enhanced['Day'] = df_enhanced['Time'] // (3600 * 24)
        
        # Amount-based features
        df_enhanced['Log_Amount'] = np.log1p(df_enhanced['Amount'])
        df_enhanced['Amount_Squared'] = df_enhanced['Amount'] ** 2
        
        # Interaction features (example with some V features)
        if 'V1' in df.columns and 'V2' in df.columns:
            df_enhanced['V1_V2_interaction'] = df_enhanced['V1'] * df_enhanced['V2']
        
        return df_enhanced
    
    def get_feature_statistics(self, df, target_col='Class'):
        """
        Get statistical summary of features by class.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            Dictionary with statistics for each class
        """
        stats = {}
        
        for class_value in df[target_col].unique():
            class_data = df[df[target_col] == class_value]
            stats[class_value] = {
                'count': len(class_data),
                'mean_amount': class_data['Amount'].mean(),
                'median_amount': class_data['Amount'].median(),
                'std_amount': class_data['Amount'].std(),
                'min_amount': class_data['Amount'].min(),
                'max_amount': class_data['Amount'].max()
            }
        
        return stats


def load_and_validate_data(file_path, required_columns=None):
    """
    Load and validate credit card transaction data.
    
    Args:
        file_path (str): Path to the CSV file
        required_columns (list): List of required column names
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If file doesn't exist
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
        print(df.isnull().sum())
    
    return df


def split_features_target(df, target_col='Class'):
    """
    Split dataframe into features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        tuple: (X, y) features and target
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Utilities for Credit Card Fraud Detection")
    print("\nAvailable preprocessing methods:")
    print("1. Feature Scaling (Standard, MinMax, Robust)")
    print("2. Class Imbalance Handling (SMOTE, ADASYN, Under-sampling, SMOTE-Tomek)")
    print("3. Outlier Removal")
    print("4. Feature Engineering")
    print("5. Statistical Analysis")
