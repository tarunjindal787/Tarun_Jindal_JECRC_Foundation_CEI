import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import pickle
import os
from data_pipeline import load_data, get_city_wide_daily

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def extract_time_features(df, datetime_col='date'):
    df = df.copy()
    df['date'] = pd.to_datetime(df[datetime_col])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def train_forecasting_model():
    print("Loading data for forecasting...")
    df = load_data()
    city_daily = get_city_wide_daily(df)
    
    print("Training Forecasting Model...")
    # Feature engineering
    city_daily = extract_time_features(city_daily, 'date')
    
    # We want to predict energy(kWh/hh)
    features = ['dayofweek', 'month', 'dayofyear', 'is_weekend']
    X = city_daily[features]
    y = city_daily['energy(kWh/hh)']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "forecast_model.pkl"), 'wb') as f:
        pickle.dump(rf, f)
    
    # Save the latest date for reference during prediction
    with open(os.path.join(MODEL_DIR, "latest_date.txt"), 'w') as f:
        f.write(city_daily['date'].max().strftime('%Y-%m-%d'))
        
    print("Forecasting model trained and saved.")

def train_anomaly_model():
    print("Loading data for anomaly detection...")
    df = load_data()
    
    # Let's find anomalies at the household level on daily aggregates
    df['date'] = df['tstp'].dt.date
    daily_hh = df.groupby(['date', 'LCLid'])['energy(kWh/hh)'].sum().reset_index()
    
    print("Training Anomaly Detection Model...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    
    X = daily_hh[['energy(kWh/hh)']]
    iso_forest.fit(X)
    
    with open(os.path.join(MODEL_DIR, "anomaly_model.pkl"), 'wb') as f:
        pickle.dump(iso_forest, f)
        
    print("Anomaly detection model trained and saved.")

if __name__ == "__main__":
    train_forecasting_model()
    train_anomaly_model()
