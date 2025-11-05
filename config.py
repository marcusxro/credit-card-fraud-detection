"""
Configuration file for Credit Card Fraud Detection project
"""

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'creditcard.csv',
    'test_size': 0.3,
    'random_state': 42,
    'stratify': True
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'use_smote': True,
    'smote_random_state': 42,
    'scaling_method': 'standard',  # 'standard', 'minmax', or 'robust'
}

# Model Configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs',
        'class_weight': None  # Can be 'balanced' or None
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': None  # Can be 'balanced' or None
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'cross_validation_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
}

# Output Configuration
OUTPUT_CONFIG = {
    'plots_dir': 'plots',
    'models_dir': 'saved_models',
    'results_dir': 'results'
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'default_model': 'random_forest',  # 'logistic_regression' or 'random_forest'
    'probability_threshold': 0.5,
    'risk_levels': {
        'low': (0.0, 0.3),
        'medium': (0.3, 0.6),
        'high': (0.6, 0.8),
        'critical': (0.8, 1.0)
    }
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': ['#2ecc71', '#e74c3c'],  # Normal, Fraud
    'save_format': 'png'
}
