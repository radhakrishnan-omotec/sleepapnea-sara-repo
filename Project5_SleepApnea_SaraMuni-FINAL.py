# %% [markdown]
# # Enhanced Sleep Apnea Prediction Notebook
# ### Comprehensive Analysis with Feature Engineering and Advanced Modeling

# %% [markdown]
# ## Step 1: Enhanced Imports with Additional Libraries
# Import critical dependencies for data processing, visualization, modeling, and interpretation
# - pandas/numpy: Data manipulation
# - sklearn: Machine learning components
# - shap: Model explainability
# - imblearn: Handling class imbalance
# - joblib: Model serialization for production
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, 
                                  OneHotEncoder, FunctionTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            roc_auc_score, precision_recall_curve, RocCurveDisplay)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import shap
import joblib
pd.set_option('display.max_columns', 50)
plt.style.use('seaborn-darkgrid')

# %% [markdown]
# ## Step 2: Advanced Data Loading with Comprehensive Checks
# Load dataset with type specification and initial validation
# - Explicit dtype specification ensures correct parsing
# - Data validation checks for duplicates and unique identifiers
# - Shape/dtype verification provides initial data understanding
dtypes = {
    'Blood Pressure': 'object',  # Preserve string format for splitting
    'Neck Thickness (cm)': 'float64',
    'Mean Apnea–Hypopnea Duration (s)': 'float64'
}

def load_data(path):
    """Load and validate dataset with critical integrity checks"""
    df = pd.read_csv(path, dtype=dtypes)
    
    # Data quality checks
    if df.duplicated().sum() > 0:
        raise ValueError("Duplicate records found in dataset")
    if not df['Person ID'].is_unique:
        raise ValueError("Non-unique patient IDs detected")
    
    print(f"Dataset Shape: {df.shape}")
    print("\nInitial Data Types:")
    print(df.dtypes.value_counts())
    return df

df = load_data("enhanced_full_sleep_apnea_dataset.csv")

# %% [markdown]
# ## Step 3: Enhanced EDA with Automated Profiling
# Comprehensive exploratory analysis combining automated reporting and custom visualizations
# - pandas_profiling generates interactive HTML report
# - Custom visualizations highlight key clinical relationships
# - Blood pressure analysis reveals hypertension patterns
from pandas_profiling import ProfileReport

def generate_eda_profile(df):
    """Generate interactive HTML report with automated analysis"""
    profile = ProfileReport(df, title="Sleep Apnea Dataset Profiling", explorative=True)
    profile.to_file("sleep_apnea_profile.html")

def create_custom_visualizations(df):
    """Generate clinical insight-focused visualizations"""
    plt.figure(figsize=(18, 12))
    
    # Target distribution analysis
    plt.subplot(2, 2, 1)
    sns.countplot(x='Sleep Disorder', data=df, palette='viridis')
    plt.title('Target Class Distribution')
    
    # Age-disorder relationship
    plt.subplot(2, 2, 2)
    sns.violinplot(x='Sleep Disorder', y='Age', data=df, palette='magma')
    plt.title('Age Distribution by Disorder')
    
    # Blood pressure analysis
    plt.subplot(2, 2, 3)
    bp = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    bp.plot(kind='hexbin', gridsize=15, title='Blood Pressure Distribution',
            xlabel='Systolic', ylabel='Diastolic')
    
    # Feature correlation
    plt.subplot(2, 2, 4)
    sns.heatmap(df.select_dtypes(include=np.number).corr(), 
                annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    return plt

generate_eda_profile(df)
eda_plots = create_custom_visualizations(df)
eda_plots.show()

# %% [markdown]
# ## Step 4: Advanced Feature Engineering
# Domain-specific feature creation for enhanced clinical relevance:
# - Blood pressure decomposition into systolic/diastolic
# - Mean Arterial Pressure (MAP) calculation for cardiovascular risk
# - Breathing difficulty composite score combining anatomical factors
# - Risk factor aggregation for holistic risk assessment
def create_features(df):
    """Engineer clinically relevant features through:
    1. Blood pressure decomposition
    2. Physiological score calculations
    3. Risk factor aggregation"""
    
    # Split blood pressure into components
    bp = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df['Systolic'] = bp[0]  # Cardiac contraction pressure
    df['Diastolic'] = bp[1]  # Cardiac relaxation pressure
    
    # Calculate Mean Arterial Pressure (MAP)
    df['MAP'] = df['Diastolic'] + 0.33*(df['Systolic'] - df['Diastolic'])
    
    # Create breathing difficulty score
    tongue_size_map = {'Normal':1, 'Large':2, 'Very Large':3}
    df['Breathing_Score'] = (
        df['Neck Thickness (cm)'] * 0.3 +  # Neck circumference contribution
        df['Tongue Size'].map(tongue_size_map) * 0.7  # Tongue obstruction risk
    )
    
    # Aggregate risk factors
    risk_factors = ['Family History of Sleep Apnea', 'Past Stroke',
                   'Type 2 Diabetes', 'Smoking Habit']
    df['Total_Risk_Factors'] = df[risk_factors].sum(axis=1)
    
    return df.drop('Blood Pressure', axis=1)  # Remove original column

df = create_features(df)

# %% [markdown]
# ## Step 5: Advanced Preprocessing Pipeline
# Robust data preparation pipeline handling:
# - Numeric features: Imputation + scaling
# - Categorical features: One-hot encoding
# - Ordinal features: Preserve inherent order
# - Column-wise processing for mixed data types
def build_preprocessor():
    """Create comprehensive data processing pipeline:
    Numeric: Median imputation + standardization
    Categorical: Mode imputation + one-hot encoding
    Ordinal: Ordered encoding for stress/sleep quality"""
    
    num_features = ['Age', 'Sleep Duration', 'Physical Activity Level',
                   'Daily Steps', 'Mean Apnea–Hypopnea Duration (s)',
                   'Neck Thickness (cm)', 'Systolic', 'Diastolic', 'MAP']
    cat_features = ['Gender', 'Occupation', 'BMI Category',
                   'Common Nasal Congestion Causes', 'Tongue Size']
    ordinal_features = ['Stress Level', 'Quality of Sleep']
    
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Robust to outliers
        ('scaler', StandardScaler())  # Standardize feature scales
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categories
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))  # Expand categoricals
    ])
    
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[[1,2,3,4,5,6,7,8,9,10],  # Stress levels
                              [1,2,3,4,5,6,7,8,9]]))  # Sleep quality
    ])
    
    return ColumnTransformer([
        ('num', numeric_pipe, num_features),
        ('cat', categorical_pipe, cat_features),
        ('ord', ordinal_pipe, ordinal_features)
    ], remainder='passthrough')  # Preserve engineered features

preprocessor = build_preprocessor()

# %% [markdown]
# ## Step 6: Advanced Class Balancing
# Address imbalanced classes using SMOTE:
# - Generates synthetic minority class samples
# - Preserves data distribution characteristics
# - Stratified splitting maintains class ratios
def balance_classes(X, y):
    """Apply Synthetic Minority Oversampling (SMOTE):
    1. Creates synthetic samples for minority classes
    2. Maintains data distribution patterns
    3. Uses k-NN with 5 neighbors for sample generation"""
    
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(preprocessor.fit_transform(X), y)
    return train_test_split(
        X_res, y_res, 
        test_size=0.2, 
        stratify=y_res,  # Preserve class distribution
        random_state=42
    )

X = df.drop(['Sleep Disorder', 'Person ID'], axis=1)
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = balance_classes(X, y)

# %% [markdown]
# ## Step 7: Enhanced Model Training with 5 Classifiers
# Comprehensive model selection covering different algorithm types:
# 1. Ensemble Methods (Random Forest, XGBoost, CatBoost, HistGradientBoosting)
# 2. Simple Baseline (Decision Tree)
def train_models(X_train, y_train):
    """Train and optimize five diverse classifiers:
    - Tree-based ensembles for high performance
    - Simple decision tree as baseline
    - Automatic class weighting for imbalance"""
    
    models = {
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',  # Adjust for class imbalance
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced',  # Balance class weights
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            tree_method='hist',  # Optimized histogram-based splitting
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        ),
        'CatBoost': CatBoostClassifier(
            auto_class_weights='Balanced',  # Automatic imbalance handling
            verbose=0,  # Silent training
            random_state=42
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            categorical_features=[False]*len(num_features) + 
            [True]*(len(cat_features)+len(ordinal_features)),
            random_state=42
        )
    }

    # Expanded parameter grids for comprehensive search
    params = {
        'Random Forest': {
            'n_estimators': [100, 200],  # Number of trees
            'max_depth': [None, 5, 10],  # Tree complexity control
            'min_samples_split': [2, 5]  # Splitting threshold
        },
        'Decision Tree': {
            'max_depth': [5, 10, None],  # Depth control
            'min_samples_split': [2, 5],  # Split requirements
            'criterion': ['gini', 'entropy']  # Split quality measures
        },
        'XGBoost': {
            'learning_rate': [0.01, 0.1],  # Shrinkage rate
            'max_depth': [3, 5],  # Tree depth
            'n_estimators': [100, 200],  # Boosting rounds
            'subsample': [0.8, 1.0]  # Stochastic training
        },
        'CatBoost': {
            'iterations': [100, 200],  # Total trees
            'depth': [4, 6],  # Tree complexity
            'learning_rate': [0.01, 0.1],  # Step size
            'l2_leaf_reg': [1, 3]  # Regularization
        },
        'HistGradientBoosting': {
            'max_iter': [100, 200],  # Boosting iterations
            'learning_rate': [0.1, 0.01],  # Shrinkage
            'max_depth': [3, 5],  # Tree depth
            'min_samples_leaf': [10, 20]  # Regularization
        }
    }

    best_models = {}
    for name, model in models.items():
        gs = GridSearchCV(
            model, 
            params[name],
            cv=StratifiedKFold(3),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        
        print(f"\n{name} Optimization Results:")
        print(f"Best Parameters: {gs.best_params_}")
        print(f"Best CV Score: {gs.best_score_:.3f}")
        print(f"Training Time: {gs.refit_time_:.1f}s")
    
    return best_models

best_models = train_models(X_train, y_train)

# %% [markdown]
# ## Step 8: Comprehensive Model Evaluation
# Multi-faceted performance assessment:
# - Classification metrics (precision, recall, F1)
# - ROC curves for multi-class discrimination
# - Probability calibration analysis
def evaluate_models(models, X_test, y_test):
    """Perform detailed model evaluation:
    1. Classification report with key metrics
    2. ROC curve visualization
    3. Probability distribution analysis"""
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC Curve visualization
        plt.figure()
        RocCurveDisplay.from_estimator(
            model, X_test, y_test,
            multi_class='ovo',  # One-vs-One strategy
            name=name
        )
        plt.title(f'{name} ROC Curves')
        plt.show()

evaluate_models(best_models, X_test, y_test)

# %% [markdown]
# ## Step 9: Model Interpretation with SHAP
# Explain model predictions using SHAP values:
# - Global feature importance
# - Individual prediction explanations
# - Interaction effects analysis
def explain_model(model, X_test, feature_names, class_names):
    """Generate model explanations using SHAP:
    1. Feature importance rankings
    2. Individual prediction breakdowns
    3. Interaction effect visualization"""
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        class_names=class_names,
        plot_type='bar'  # Global feature importance
    )
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()

feature_names = preprocessor.get_feature_names_out()
class_names = df['Sleep Disorder'].unique()
explain_model(best_models['XGBoost'], X_test, feature_names, class_names)

# %% [markdown]
# ## Step 10: Model Deployment Preparation
# Production-ready pipeline packaging:
# - Unified preprocessing + modeling pipeline
# - Model serialization for deployment
# - Example inference implementation
def deploy_model(preprocessor, model):
    """Create production pipeline and save artifacts:
    1. Combine preprocessing and modeling steps
    2. Serialize with joblib for portability
    3. Create sample inference demonstration"""
    
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Save complete pipeline
    joblib.dump(final_pipe, 'sleep_apnea_pipeline.pkl')
    
    # Demonstrate inference
    sample = X.sample(1)
    print("\nSample Prediction:")
    print(f"Input Features:\n{sample}")
    print(f"Predicted Class: {final_pipe.predict(sample)[0]}")
    print(f"Class Probabilities: {final_pipe.predict_proba(sample)[0]}")
    
    return final_pipe

production_pipeline = deploy_model(preprocessor, best_models['XGBoost'])