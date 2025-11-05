CREDIT CARD FRAUD DETECTION SYSTEM
===================================

DESCRIPTION:
------------
This project detects fraudulent transactions using machine learning. 
It uses a dataset from Kaggle containing anonymized credit card transactions 
and classifies whether a transaction is fraudulent or legitimate.

SETUP INSTRUCTIONS:
-------------------

1. CREATE VIRTUAL ENVIRONMENT
-----------------------------
Create a Python virtual environment to keep dependencies organized.

> python -m venv venv
> venv\Scripts\activate      (for Windows)
> source venv/bin/activate   (for macOS/Linux)


2. INSTALL REQUIREMENTS
-----------------------
Install the necessary libraries using the requirements file.

> pip install -r requirements.txt

3. DOWNLOAD THE DATASET
-----------------------
Run the dataset script to automatically download the dataset 
from Kaggle and place it into the `/data` folder.

> python dataset.py

This script fetches the "Credit Card Fraud Detection" dataset 
and extracts it to be ready for model training.


4. TRAIN THE MODEL
------------------
Run the training script to process the data, train the model, 
and save both the trained model and scaler to the `/models` folder.

> python train_model.py

After completion, two files will be created in the `models` folder:
- fraud_model.pkl
- scaler.pkl


5. TEST THE MODEL
-----------------
Run the testing script to evaluate the model performance on sample data 
or a testing dataset. This will display predictions, accuracy, and recall.

> python test_model.py

This step confirms whether the trained model can accurately 
distinguish between legitimate and fraudulent transactions.


NOTES:
------
- Always run scripts while the virtual environment is active.
- Do not commit large dataset files into the repository.
- If you modify dependencies, update `requirements.txt` by running:
  > pip freeze > requirements.txt
- The dataset file is usually named `creditcard.csv` and should be located in the `/data` directory.
