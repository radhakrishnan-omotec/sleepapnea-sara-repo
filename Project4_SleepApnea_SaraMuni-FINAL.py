# Google Colab Compatible Sleep Apnea Analysis Notebook

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# Step 2: Load Dataset from Google Drive or Local Upload
df = pd.read_csv("/content/Sleep_health_and_lifestyle_dataset.csv")

# Step 3: Display Dataset Information
print("Dataset Overview:")
df.info()
print("\nFirst 5 Rows:")
display(df.head())

# Step 4: Handle Missing Values
df.fillna(df.median(), inplace=True)

# Step 5: Encode Categorical Data
label_encoders = {}
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Tongue Size', 'Use of Muscle-Relaxing Substances', 'Common Nasal Congestion Causes', 'Family History of Sleep Apnea', 'Past Stroke', 'Type 2 Diabetes']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Feature Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 7: Train-Test Split
X = df.drop(columns=['Sleep Disorder'])  # Features
y = df['Sleep Disorder']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8a: Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg) * 100:.2f}%")

# Step 8b: XGBoost Classifier Model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")

# Step 8c: CatBoost Classifier Model
catboost_model = CatBoostClassifier(verbose=False)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_catboost) * 100:.2f}%")

# Step 8d: Gradient Boosting Classifier Model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb) * 100:.2f}%")

# Step 8e: Support Vector Classifier (SVC) Model
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
print(f"SVC Accuracy: {accuracy_score(y_test, y_pred_svc) * 100:.2f}%")

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 10: Model Interpretation using SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
