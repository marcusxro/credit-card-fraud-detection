# Credit Card Fraud Detection 💳🔍

A comprehensive machine learning project for detecting fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset. This project implements multiple ML algorithms with extensive data analysis, visualization, and model evaluation capabilities.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Credit card fraud is a significant problem in financial transactions. This project uses machine learning to identify fraudulent transactions from a highly imbalanced dataset. The solution includes:

- **Data preprocessing** with standardization and SMOTE for handling class imbalance
- **Multiple ML models** including Logistic Regression and Random Forest
- **Comprehensive evaluation** with various metrics and visualizations
- **Prediction system** for real-time fraud detection
- **Interactive Jupyter notebook** for exploratory data analysis

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle:

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent transactions**: 492 (0.172% of all transactions)
- **Features**: 30 numerical features (Time, V1-V28, Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)

### Dataset Characteristics

- **Highly imbalanced**: Only 0.172% of transactions are fraudulent
- **PCA transformed**: Features V1-V28 are principal components (for confidentiality)
- **Time**: Seconds elapsed between each transaction and the first transaction
- **Amount**: Transaction amount (useful for cost-sensitive learning)

### Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place the file in the project root directory

## ✨ Features

### Data Analysis & Visualization
- 📈 Comprehensive exploratory data analysis
- 📊 Class distribution analysis
- 🔍 Feature correlation analysis
- 📉 Transaction amount and time distribution plots
- 🎨 Correlation heatmaps

### Machine Learning Models
- 🤖 Logistic Regression
- 🌲 Random Forest Classifier
- ⚖️ SMOTE for handling class imbalance
- 📏 StandardScaler for feature normalization

### Model Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- 📈 ROC-AUC Score and ROC Curves
- 📊 Confusion Matrices
- 🔄 Cross-validation
- 📉 Precision-Recall Curves

### Prediction System
- 🎯 Single transaction prediction
- 📦 Batch prediction from CSV files
- ⚠️ Risk level classification (Low, Medium, High, Critical)
- 💾 Model persistence with joblib

### Interactive Tools
- 📓 Jupyter notebook for interactive exploration
- 🔧 Configuration file for easy parameter tuning
- 📁 Organized output structure (plots, models, results)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/marcusxro/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place it in the project root directory

## 📖 Usage

### 1. Train the Models

Run the main training script to train all models, generate visualizations, and save the trained models:

```bash
python fraud_detection.py
```

This will:
- Load and analyze the dataset
- Create visualizations in the `plots/` directory
- Preprocess data with SMOTE
- Preprocess data with SMOTE
- Train Logistic Regression and Random Forest models
- Evaluate models with comprehensive metrics
- Save trained models to `saved_models/` directory

**Output**:
- `plots/class_distribution.png` - Class distribution visualization
- `plots/amount_distribution.png` - Transaction amount analysis
- `plots/time_distribution.png` - Transaction time analysis
- `plots/correlation_heatmap.png` - Feature correlation heatmap
- `plots/model_comparison.png` - Model performance comparison
- `plots/roc_curves.png` - ROC curves for all models
- `plots/confusion_matrices.png` - Confusion matrices
- `saved_models/*.pkl` - Trained models and scaler

### 2. Make Predictions

#### Single Prediction (Interactive Mode)
```bash
python predict.py
```

#### Batch Predictions from CSV
```bash
python predict.py transactions.csv
```

This will:
- Load the trained model
- Make predictions on all transactions in the CSV
- Generate risk levels for each transaction
- Save results to `transactions_predictions.csv`
- Display high-risk transactions

### 3. Interactive Analysis with Jupyter Notebook

Launch the Jupyter notebook for interactive exploration:

```bash
jupyter notebook fraud_detection_notebook.ipynb
```

The notebook includes:
- Step-by-step data exploration
- Interactive visualizations
- Model training and evaluation
- Example predictions

### 4. Custom Configuration

Modify `config.py` to customize:
- Model hyperparameters
- Train-test split ratio
- SMOTE settings
- Output directories
- Visualization preferences

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py          # Main training script
├── predict.py                  # Prediction script
├── config.py                   # Configuration file
├── fraud_detection_notebook.ipynb  # Interactive Jupyter notebook
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
│
├── creditcard.csv              # Dataset (download separately)
│
├── plots/                      # Generated visualizations
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── confusion_matrices.png
│
└── saved_models/               # Trained models
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    └── scaler.pkl
```

## 📊 Model Performance

The models are evaluated using multiple metrics to handle the imbalanced dataset:

### Logistic Regression
- **Accuracy**: ~99.9%
- **Precision**: High precision in detecting fraud
- **Recall**: Good recall for fraud detection
- **F1-Score**: Balanced performance
- **ROC-AUC**: Excellent separation capability

### Random Forest Classifier
- **Accuracy**: ~99.9%
- **Precision**: Superior fraud detection precision
- **Recall**: Excellent fraud recall
- **F1-Score**: Best overall balance
- **ROC-AUC**: Outstanding performance

> **Note**: Due to the highly imbalanced nature of the dataset, accuracy alone is not a reliable metric. We focus on Precision, Recall, F1-Score, and ROC-AUC for proper evaluation.

### Key Insights
- 🎯 Random Forest typically outperforms Logistic Regression
- ⚖️ SMOTE significantly improves model performance on fraud detection
- 📈 Both models achieve excellent ROC-AUC scores (>0.95)
- 🎲 Very few false positives while maintaining high fraud detection rate

## 🎨 Visualizations

The project generates comprehensive visualizations:

1. **Class Distribution**: Shows the severe imbalance in the dataset
2. **Amount Distribution**: Compares transaction amounts for normal vs. fraud
3. **Time Distribution**: Analyzes when fraudulent transactions occur
4. **Correlation Heatmap**: Identifies features most correlated with fraud
5. **Model Comparison**: Side-by-side comparison of all metrics
6. **ROC Curves**: Visual comparison of model performance
7. **Confusion Matrices**: Detailed breakdown of predictions

All visualizations are automatically saved to the `plots/` directory.

## 🛠️ Technologies Used

### Core Libraries
- **NumPy** (1.24.3) - Numerical computing
- **Pandas** (2.0.3) - Data manipulation and analysis
- **scikit-learn** (1.3.0) - Machine learning algorithms

### Visualization
- **Matplotlib** (3.7.2) - Plotting and visualization
- **Seaborn** (0.12.2) - Statistical data visualization

### Class Imbalance Handling
- **imbalanced-learn** (0.11.0) - SMOTE and other sampling techniques

### Model Persistence
- **joblib** (1.3.2) - Efficient model serialization

### Interactive Development
- **Jupyter** (1.0.0) - Interactive notebook environment
- **notebook** (7.0.2) - Jupyter notebook interface

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- 🤖 Add more ML models (XGBoost, Neural Networks, etc.)
- 🎨 Improve visualizations
- 📊 Add more evaluation metrics
- 🔧 Enhance configuration options
- 📚 Improve documentation
- 🧪 Add unit tests
- 🌐 Create a web interface for predictions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Machine Learning Group - ULB (Université Libre de Bruxelles)
- **Dataset Publication**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
  *Calibrating Probability with Undersampling for Unbalanced Classification.* 
  In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
- **Kaggle**: For hosting the dataset and providing a platform for data science

## 📧 Contact

**Marcus** - [@marcusxro](https://github.com/marcusxro)

Project Link: [https://github.com/marcusxro/credit-card-fraud-detection](https://github.com/marcusxro/credit-card-fraud-detection)

---

⭐ If you found this project helpful, please consider giving it a star!

**Happy Fraud Detecting! 🕵️‍♂️💳**