# final_task4.py — copy this, set FILENAME to your CSV, and run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# --------------- CHANGE ONLY THIS ----------------
FILENAME = "C:\\Users\\G HARSHITHA\\Downloads\\archive (5).zip"   # <- replace with your CSV filename (example: "breast_cancer.csv")
# --------------------------------------------------

# 1. Load
df = pd.read_csv(FILENAME)
print("Loaded. Columns:", list(df.columns))

# 2. Drop obvious useless columns if present
for col in ["Unnamed: 32", "id", "Id", "ID"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
        print(f"Dropped column: {col}")

# 3. Drop rows with missing values (or you can impute if preferred)
n_before = len(df)
df = df.dropna()
print(f"Dropped {n_before - len(df)} rows with missing values. Remaining rows: {len(df)}")

# 4. Detect target column (prefer 'diagnosis' if exists)
if "diagnosis" in df.columns:
    target_col = "diagnosis"
else:
    # find a binary column automatically
    target_col = None
    for c in df.columns:
        if df[c].nunique() == 2:
            target_col = c
            break
    if target_col is None:
        raise SystemExit("No binary target column found. Print df.columns and pick the target.")

print("Detected target column:", target_col)
print("Target value counts:\n", df[target_col].value_counts())

# 5. Prepare X and y
y = df[target_col].copy()
X = df.drop(columns=target_col)

# 6. Encode non-numeric target to 0/1 (handles 'B'/'M' etc.)
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Encoded target classes:", dict(enumerate(le.classes_)))  # e.g. {0: 'B', 1: 'M'}

# 7. Keep only numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) != X.shape[1]:
    dropped = list(set(X.columns) - set(numeric_cols))
    print("Dropping non-numeric feature columns:", dropped)
X = X[numeric_cols]

if X.shape[1] == 0:
    raise SystemExit("No numeric features left. Check your CSV.")

# 8. Train-test split (stratify to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Train Logistic Regression
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

# 11. Predict & Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# 12. Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 13. Threshold tuning example
threshold = 0.6
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\nConfusion Matrix with threshold={threshold}:\n", confusion_matrix(y_test, y_pred_custom))
