import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import json

# Load and preprocess the data
df = pd.read_csv("data/cleaned_dataset.csv")
df['price_per_square_meter'] = df['price'] / df['total_area_sqm']

# Handle outliers based on price per square meter for each location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('zip_code'):
        m = np.mean(subdf.price_per_square_meter)
        st = np.std(subdf.price_per_square_meter)
        reduced_df = subdf[(subdf.price_per_square_meter > (m - st)) & (subdf.price_per_square_meter <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)
df['zip_code'] = df['zip_code'].astype(str)
dummies = pd.get_dummies(df.zip_code)
df = pd.concat([df, dummies], axis=1)
df = df.drop('zip_code', axis=1)

# Map EPC to numeric values
epc_mapping = {
    'A++': 9, 'A+': 8, 'A': 7, 'B': 6,
    'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1
}
df['epc'] = df['epc'].map(epc_mapping)

# Prepare features and target
X = df.drop(['price', 'price_per_square_meter'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best parameters from previous GridSearchCV run
best_params = {
    'colsample_bytree': 1.0,
    'gamma': 0,
    'learning_rate': 0.1, #lowered from 0.2
    'max_depth': 5,
    'n_estimators': 500, # 300
    'reg_alpha': 0.1, #0
    'reg_lambda': 1.5, #1
    'subsample': 0.6
}

# Initialize the XGBoost model with best parameters and GPU configuration
best_xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',  # Enables GPU usage
    device="cuda",
    **best_params
)

# Train the model
best_xgb_model.fit(X_train, y_train)

# Evaluate the model
print("XGBoost Training Score:", best_xgb_model.score(X_train, y_train))
print("XGBoost Test Score:", best_xgb_model.score(X_test, y_test))

# Cross-validation for further evaluation
xgb_cross_val_scores = cross_val_score(best_xgb_model, X, y, cv=10)
print("Mean Cross-Validation Score for XGBoost:", xgb_cross_val_scores.mean())

# Make predictions using the trained model
y_pred = best_xgb_model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error for test (MAE):", mae)
