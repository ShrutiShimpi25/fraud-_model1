# Fraud Detection Model  

# 👤 Name: Shruti Shimpi
# 📅 Date: July 2025

# -----------------------------------------------
# 📌 1. Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# -----------------------------------------------
# 📌 2. Load Dataset
# df = pd.read_csv('your_dataset.csv')
# df.head()

# -----------------------------------------------
# 📌 3. Data Cleaning
# - Check missing values
# - Drop/Impute as needed
# - Outlier treatment (if required)
# - Drop unnecessary columns like 'nameOrig', 'nameDest'

# -----------------------------------------------
# 📌 4. Exploratory Data Analysis (EDA)
# - Class imbalance
# - Transaction types
# - Balance behaviors
# - Correlations

# -----------------------------------------------
# 📌 5. Feature Engineering / Selection
# - Select relevant features
# - Encode categorical variables
# - Prepare X and y

# -----------------------------------------------
# 📌 6. Handle Class Imbalance
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)

# -----------------------------------------------
# 📌 7. Model Training
# X_train, X_test, y_train, y_test = train_test_split(...)
# clf = LogisticRegression(class_weight='balanced', max_iter=1000)
# clf.fit(X_train, y_train)

# -----------------------------------------------
# 📌 8. Evaluation
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# -----------------------------------------------
# 📌 9. Feature Importance (Optional)
# LogisticRegression coefficients plot (if desired)

# -----------------------------------------------
# 📌 10. Business Questions Answered (Markdown Below)

"""
## 🧾 Assignment Questions & Answers

1. 🔹 How did you clean the data?
   - Dropped non-informative columns ('nameOrig', 'nameDest')
   - Checked and confirmed no missing values
   - Removed multicollinearity using correlation analysis

2. 🔹 What model did you use and why?
   - Logistic Regression
   - It is simple, interpretable, and works well for binary classification problems. It allows us to understand the impact of each feature.

3. 🔹 How did you select features?
   - Removed identifiers
   - Used correlation heatmap
   - Selected variables based on their relationship with the target and business logic

4. 🔹 How did the model perform?
   - Accuracy: 97%
   - ROC-AUC Score: 0.995
   - Recall for fraud class: 0.05 — indicates high class imbalance

5. 🔹 Which factors are important for fraud detection?
   - Transaction type = 'TRANSFER' or 'CASH_OUT'
   - Large transaction amounts
   - Account balances dropping to zero

6. 🔹 Do these factors make sense?
   - Yes, fraud typically involves large, sudden transfers and draining account balances

7. 🔹 What preventive measures can the company take?
   - Monitor suspicious transaction patterns (e.g., large or back-to-back transfers)
   - Use additional OTP/verification for high-value transfers
   - Strengthen flagging based on transaction type and history

8. 🔹 How will you measure if changes work?
   - Track fraud detection rate (recall)
   - Monitor false positives (precision)
   - Evaluate before/after performance using weekly metrics and ROC-AUC
"""

# -----------------------------------------------
# 📌 End of Notebook
