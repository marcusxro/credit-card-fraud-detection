from inference import FraudDetector
import pandas as pd
import numpy as np

detector = FraudDetector()

df = pd.read_csv('data/creditcard.csv')

print(f"Testing {len(df)} transactions from creditcard\n")

transactions = df.drop('Class', axis=1) if 'Class' in df.columns else df
actual_labels = df['Class'].tolist() if 'Class' in df.columns else None

predictions = detector.predict(transactions)

expected_cols = list(detector.scaler.feature_names_in_)
rename_map = {col.lower(): col for col in expected_cols}
X_processed = transactions.rename(columns=rename_map)
X_processed[expected_cols] = detector.scaler.transform(X_processed[expected_cols])
probabilities = detector.model.predict_proba(X_processed)

correct = 0
fraud_detected = 0
fraud_actual = 0

for i in range(len(predictions)):
    pred = predictions[i]
    prob = probabilities[i][1]
    
    if actual_labels:
        actual = actual_labels[i]
        match = "MATCH" if pred == actual else "MISS"
        correct += (pred == actual)
        fraud_actual += actual
    else:
        match = "N/A"
        actual = "?"
    
    fraud_detected += pred
    status = "FRAUD" if pred == 1 else "LEGIT"
    
    print(f"Trans {i+1:3d}: {status:5s} | Prob: {prob:6.2%} | Actual: {actual} | {match}")

print(f"\n{'='*50}")
print(f"Total transactions: {len(predictions)}")
print(f"Predicted fraud: {fraud_detected}")

if actual_labels:
    print(f"Actual fraud: {fraud_actual}")
    print(f"Accuracy: {correct}/{len(predictions)} ({correct/len(predictions)*100:.1f}%)")
    
    tp = sum(1 for p, a in zip(predictions, actual_labels) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(predictions, actual_labels) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(predictions, actual_labels) if p == 0 and a == 1)
    
    print(f"\nTrue Positives (caught fraud): {tp}")
    print(f"False Positives (false alarm): {fp}")
    print(f"False Negatives (missed fraud): {fn}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.2%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.2%}")