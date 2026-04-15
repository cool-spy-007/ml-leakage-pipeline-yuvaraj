import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset Generation
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# --- Task 1: Reproduce and Identify Leakage ---
print("--- Task 1: Flawed Approach (Leakage) ---")

# Scaling BEFORE splitting (The Leakage)
scaler = StandardScaler()
X_scaled_flawed = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_flawed, y, test_size=0.2, random_state=42)

model_flawed = LogisticRegression()
model_flawed.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model_flawed.predict(X_train))
test_acc = accuracy_score(y_test, model_flawed.predict(X_test))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("Problem: Data leakage occurred because the scaler 'saw' the distribution of the entire dataset (including the test set) before the split. This informs the training process about the test set's mean and variance, inflating performance metrics.")


# --- Task 2: Fix the Workflow Using a Pipeline ---
print("\n--- Task 2: Corrected Pipeline Approach ---")

# Split FIRST
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# 5-fold Cross-Validation
cv_scores = cross_val_score(pipeline, X_train_clean, y_train_clean, cv=5)

print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")
print(f"CV Standard Deviation: {cv_scores.std():.4f}")


# --- Task 3: Experiment with Decision Tree Depth ---
print("\n--- Task 3: Decision Tree Experiment ---")

depths = [1, 5, 20]
results = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train_clean, y_train_clean)
    
    tr_acc = accuracy_score(y_train_clean, dt.predict(X_train_clean))
    ts_acc = accuracy_score(y_test_clean, dt.predict(X_test_clean))
    
    results.append({"Max Depth": d, "Train Accuracy": round(tr_acc, 4), "Test Accuracy": round(ts_acc, 4)})

# Display Table
df_results = pd.DataFrame(results)
print(df_results)

print("\nAnalysis:")
print("The depth of 5 best balances fit and generalization. Depth 1 underfits (low accuracy on both), while depth 20 overfits (perfect training accuracy but lower test performance compared to depth 5).")