from feature_engineering import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Use RandomForestRegressor as XGBoost/LightGBM might not be pre-installed in all environments 
# (though it usually is on Kaggle). We'll use sklearn's standard ensemble for robust flat execution.
from sklearn.ensemble import RandomForestRegressor

print("--- RUNNING MODEL TRAINING ---")

# Define feature columns to use for training
feature_cols = [
    'energy_lag_1', 'energy_lag_7', 'energy_rolling_7', 
    'day_of_week', 'month', 'is_weekend', 'is_holiday',
    'temperatureMax', 'temperatureMin', 'windSpeed'
]

# Ensure all selected feature columns exist (in case of dummy data lacking some weather columns)
available_features = [c for c in feature_cols if c in merged_df.columns]

# Split the final merged dataframe chronologically into train and test splits based on a boundary date
print("Splitting dataset chronologically...")
# Use the 80th percentile date as the boundary
boundary_date = merged_df['day'].quantile(0.8)

train_df = merged_df[merged_df['day'] < boundary_date].copy()
test_df = merged_df[merged_df['day'] >= boundary_date].copy()

# Fill any remaining NaNs in features (e.g. if weather was missing completely)
train_df[available_features] = train_df[available_features].fillna(0)
test_df[available_features] = test_df[available_features].fillna(0)

X_train = train_df[available_features]
y_train = train_df['energy_sum']

X_test = test_df[available_features]
y_test = test_df['energy_sum']

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# Initialize a tabular forecasting model, fit it to the training arrays
print("Training regressor model...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions on the test set
print("Generating predictions...")
predictions = model.predict(X_test)

# Print evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print("\n=== MODEL EVALUATION METRICS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print("================================\n")

print("Pipeline execution completed successfully.")
