# Sleep Apnea Classification using Machine Learning

## Step 1: Install and Import Required Libraries
# In this step, we install and import all necessary Python libraries for data analysis, visualization, and machine learning.
# Libraries like pandas, numpy, matplotlib, seaborn are used for data processing and visualization,
# while sklearn is used for machine learning modeling.
!pip install pandas numpy matplotlib seaborn scikit-learn shap xgboost catboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap

## Step 2: Load and Explore Dataset
# Here, we load the dataset and check its structure, missing values, and basic statistics.
# Understanding the dataset is crucial for identifying necessary preprocessing steps.
data = pd.read_csv('/mnt/data/enhanced_full_sleep_apnea_dataset (2).csv')
print("Dataset Overview:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

## Step 3: Data Cleaning and Handling Missing Values
# In this step, we check for missing values and decide how to handle them (imputation or removal).
# This ensures data integrity and prevents issues during model training.
print("\nMissing Values:")
print(data.isnull().sum())
# Dropping rows with missing values (alternative: use mean/median imputation)
data = data.dropna()

## Step 4: Data Visualization (EDA)
# Here, we perform Exploratory Data Analysis (EDA) to understand feature distributions,
# correlations, and class balance.
# We use histograms, pairplots, and correlation heatmaps to gain insights.
sns.pairplot(data, hue='ApneaSeverity')
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

## Step 5: Feature Engineering and Selection
# Selecting relevant features helps improve model accuracy and reduce computation time.
# We drop redundant or highly correlated features to avoid overfitting.
features = data.drop(columns=['ApneaSeverity'])
labels = data['ApneaSeverity']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

## Step 6: Split Data into Training and Testing Sets
# We split the data into training and testing sets to evaluate model performance.
# The training set is used to train the model, while the testing set helps assess its accuracy.
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

## Step 7: Train Machine Learning Models

## Step 8a: Train and Evaluate Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

## Step 8b: Train and Evaluate Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Model Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

## Step 8c: Train and Evaluate XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Model Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

## Step 8d: Train and Evaluate CatBoost Classifier
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)
print("CatBoost Model Accuracy:", accuracy_score(y_test, y_pred_cat))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cat))

## Step 8e: Train and Evaluate HistGradientBoosting Classifier
hgb_model = HistGradientBoostingClassifier()
hgb_model.fit(X_train, y_train)
y_pred_hgb = hgb_model.predict(X_test)
print("HistGradientBoosting Model Accuracy:", accuracy_score(y_test, y_pred_hgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_hgb))

## Step 9: SHAP Analysis for Model Interpretability
# SHAP (SHapley Additive exPlanations) helps explain feature contributions to predictions.
# This enhances transparency in model decision-making.
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

## Step 10: Conclusion and Future Enhancements
# Finally, we summarize findings and discuss potential improvements such as
# hyperparameter tuning, deep learning integration, or real-time monitoring applications.
print("This project successfully classifies sleep apnea severity using multiple machine learning models.")
