import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Set display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

# 1. Custom NA Handling & Data Loading
na_vals = [
    "TBD cytogenetics", "Not done", "NaN", "Non-resident of the U.S.", 
    "TBD", "N/A, F(pre-TED) not submitted", "Not tested", "Missing disease status"
]

data = pd.read_csv("TransplantPrediction/Data/train.csv", na_values=na_vals).drop(columns=['ID'])

# 2. Drop High-Missing Features (> 20% Missing)
threshold = 6000 
drop_cols = [c for c in data.columns if data[c].isna().sum() > threshold]
data_clean = data.drop(columns=drop_cols)

# Separate Target and Features
prediction = data_clean['efs']
data_features = data_clean.drop(columns=['efs', 'efs_time'])

# 3. Identify Feature Types
numeric_cols = data_features.select_dtypes(exclude=['object']).columns.tolist()
categorical_cols = data_features.select_dtypes(include=['object']).columns.tolist()

# 4. Define Preprocessor (Set sparse_output=False to ensure dense arrays)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 5. Train / Validation / Test Split
seed = 37
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    data_features, prediction, test_size=0.3, random_state=seed, stratify=prediction
)
X_test_raw, X_val_raw, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
)

# Apply Preprocessing
X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)

y_train_xgb = y_train.values.ravel()
y_val_xgb = y_val.values.ravel()
y_test_xgb = y_test.values.ravel()

# Sanity Check for Remaining NaNs
print(f"Remaining NaNs in X_train: {np.isnan(X_train).sum()}")

# 6. Model Training
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=seed,
    eval_metric=['logloss', 'auc'],
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train_xgb, 
    eval_set=[(X_val, y_val_xgb)], 
    verbose=100
)

# 7. Evaluation
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy Score: {accuracy_score(y_test_xgb, xgb_preds):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test_xgb, xgb_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_xgb, xgb_preds))

# 8. Feature Importance
plt.figure(figsize=(10, 6))
feat_names = preprocessor.get_feature_names_out()
feat_importances = pd.Series(xgb_model.feature_importances_, index=feat_names)

feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Most Important Features (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()