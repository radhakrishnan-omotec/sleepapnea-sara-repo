import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# Step 1: Load Dataset with 24 Attributes
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Step 2: Display Dataset Information
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Handle Missing Values
df.fillna(df.median(), inplace=True)

# Step 4: Encode Categorical Data
label_encoders = {}
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Tongue Size', 'Use of Muscle-Relaxing Substances', 'Common Nasal Congestion Causes', 'Family History of Sleep Apnea', 'Past Stroke', 'Type 2 Diabetes']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Feature Selection and Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 6: Train-Test Split
X = df.drop(columns=['Sleep Disorder'])  # Features
y = df['Sleep Disorder']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 10: Model Interpretation using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)