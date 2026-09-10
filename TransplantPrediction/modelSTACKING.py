import pandas as pd
import numpy as np
import os
import sidetable
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
import xgboost as xgb 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# set display options to show everything
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# load dict file
data_dict = pd.read_csv("TransplantPrediction/Data/data_dictionary.csv")
print(data_dict.head(2))

# load data file
data = pd.read_csv("TransplantPrediction/Data/train.csv")
test = pd.read_csv("TransplantPrediction/Data/test.csv")

# Take a look at the data
print(data.info()) # 28800 entries, 60 columns
print(data.head(2))

# Check the structure of the data
print(data.columns)
print(data.shape)
data.describe()

# Check for duplicate data
data.loc[data.duplicated(),] # no duplicates

# Check for null values. Inspect categorical data
data_type = data.dtypes
data_type_object = data_type[data_type == "object"]

data_object = data[data_type_object.index]

# Check categorical data
print(data_object.columns)

# Check for null values 
for c in data_object.columns:
    print("======= %s =======" % c)
    print(data_object[c].value_counts(dropna=False))
    # If you want more detailed table, use this.
    # print(data_object.stb.freq([c], cum_cols=False))
# Look through the printed categories and select anormal null values.

# Load without na values 
# Make a list of na values
na_vals = ["TBD cytogenetics","Not done","NaN","Non-resident of the U.S.","TBD","N/A, F(pre-TED) not submitted","Not tested","Missing disease status"]

# Load data file. Drop ID column as well. 
data = pd.read_csv("TransplantPrediction/Data/train.csv", na_values=na_vals).drop(columns=['ID'])

# Find how many na values each columns have and drop ones over the threshold.
threshold = 6000 # ~ 20 percent of the data
drop_index = []

for c in data.columns:
    if (data[c].isna().sum() > threshold):
        print("======= %s =======" % c)
        print(data[c].isna().value_counts(dropna=False))
        drop_index.append(c)

data_na_dropped = data.drop(columns=drop_index)

# Find how many na values each rows have 
na_row_counts = data_na_dropped.isna().sum(axis=1)
# Display the distribution of na counts per row.
na_row_counts.value_counts()

# drop rows with na values
data_na_dropped = data_na_dropped[na_row_counts <= 0] # Keep more of you want to impute
prediction = data_na_dropped['efs']
data_no_efs = data_na_dropped.drop(columns=['efs','efs_time'])

# divide into numeric data and categorical data
data_type = data_no_efs.dtypes
data_type_object_index = data_type[data_type == "object"].index
data_type_numeric_index = data_type[data_type != "object"].index

data_object = data_no_efs[data_type_object_index]
data_numeric = data_no_efs[data_type_numeric_index]

# Check each predictors to see if they are categorical values stored with numeric types
data_dict[data_dict.variable.isin(data_numeric.columns)][['variable', 'description']]
for c in data_numeric.columns:
    print("======= %s =======" % c)
    print(data_numeric[c].describe())
# Actual numeric data


# create transformers
preprocessor = ColumnTransformer(
    transformers=[
        # (name, transformer, columns)
        ('num', StandardScaler(), data_numeric.columns.tolist()),
        ('cat', OneHotEncoder(handle_unknown='ignore'), data_object.columns.tolist())
    ])


seed = 37

# split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    data_no_efs, prediction, test_size=0.3, random_state=seed
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed
)


# --- STEP 1: Process Data for non-pipeline models ---
# Fit preprocessor only on training data to avoid data leakage
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# If the output is sparse, convert to dense for PCA and DNN. 
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_val_proc = X_val_proc.toarray()
    X_test_proc = X_test_proc.toarray()

# PCA step to handle multicolinearity
pca = PCA(n_components=0.95, random_state=seed) 
X_train_pca = pca.fit_transform(X_train_proc)
X_val_pca = pca.transform(X_val_proc)
X_test_pca = pca.transform(X_test_proc)

print(f"Original features: {X_train_proc.shape[1]}")
print(f"Reduced PCA features: {X_train_pca.shape[1]}")

# Flatten target variables for XGBoost
y_train_xgb = y_train.values
y_val_xgb = y_val.values

# --- STEP 2: XGBoost ---

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,            # Helps prevent overfitting
    colsample_bytree=0.8,     # Helps with your 60+ columns
    random_state=seed,
    use_label_encoder=False,
    eval_metric=['logloss','error'],
    early_stopping_rounds=50  # Better to define here in modern XGB versions
)

xgb_model.fit(
    X_train_proc, y_train_xgb, 
    eval_set=[(X_val_proc, y_val_xgb)], 
    verbose=False
)

# --- STEP 3: Keras DNN ---
# The input shape is now the number of columns in our processed matrix
input_dim_pca = X_train_pca.shape[1]

dnn_model = keras.Sequential([
    keras.Input(shape=(input_dim_pca,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid') # Focused on efs prediction
])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,       
    restore_best_weights=True
)

dnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train DNN
dnn_model.fit(
    X_train_pca, y_train, 
    validation_data=(X_val_pca, y_val),
    epochs=50, 
    callbacks=[early_stopping],
    verbose=0
)

# --- STEP 4: SVM ---
svc_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=seed)
svc_model.fit(X_train_pca, y_train)

# --- STEP 5: The Meta-Learner (Stacking) ---

# Get probabilities from all three
# 1. SVC 
svc_val_probs = svc_model.predict_proba(X_val_pca)[:, 1]
svc_test_probs = svc_model.predict_proba(X_test_pca)[:, 1]

# 2. XGBoost (uses processed data)
xgb_val_probs = xgb_model.predict_proba(X_val_proc)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test_proc)[:, 1]

# 3. DNN 
dnn_val_probs = dnn_model.predict(X_val_pca).ravel()
dnn_test_probs = dnn_model.predict(X_test_pca).ravel()

# Combine into Meta-Features
X_val_meta = np.column_stack((svc_val_probs, xgb_val_probs, dnn_val_probs))
X_test_meta = np.column_stack((svc_test_probs, xgb_test_probs, dnn_test_probs))

# Fit logistic regression with the results from the three models above.
meta_learner = LogisticRegression()
meta_learner.fit(X_val_meta, y_val)

# Final Accuracy
final_preds = meta_learner.predict(X_test_meta)
print(f"Stacking Accuracy: {accuracy_score(y_test, final_preds):.4f}")

# See how the Meta-Learner weighs the three models
weights = meta_learner.coef_[0]
print(f"Weights -> SVC (PCA): {weights[0]:.2f}, XGB (Original): {weights[1]:.2f}, DNN (PCA): {weights[2]:.2f}")