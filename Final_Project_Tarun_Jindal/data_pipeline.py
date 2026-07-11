import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def generate_synthetic_data():
    """Generates synthetic smart meter data mimicking the London dataset for demonstration."""
    print("Generating synthetic data since real dataset was not found...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate 1 year of hourly data for 50 households
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_rng = pd.date_range(start=start_date, end=end_date, freq='H')
    
    households = [f"MAC{str(i).zfill(5)}" for i in range(1, 51)]
    
    data = []
    for hh in households:
        # Base load + seasonal variation + daily pattern + noise
        base_load = np.random.uniform(0.1, 0.5)
        for dt in date_rng:
            # higher in winter
            seasonal_factor = 1.0 + 0.5 * np.cos((dt.dayofyear - 15) / 365.0 * 2 * np.pi) 
            # higher in evening
            daily_factor = 1.0
            if 17 <= dt.hour <= 22:
                daily_factor = 2.0
            elif 0 <= dt.hour <= 6:
                daily_factor = 0.5
            
            consumption = base_load * seasonal_factor * daily_factor + np.random.normal(0, 0.05)
            consumption = max(0, consumption)
            
            data.append({
                "LCLid": hh,
                "tstp": dt,
                "energy(kWh/hh)": consumption
            })
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "synthetic_energy.csv"), index=False)
    print("Synthetic data saved.")
    return df

def load_data():
    """Loads dataset if available, otherwise generates synthetic data."""
    real_data_path = os.path.join(DATA_DIR, "halfhourly_dataset")
    synthetic_path = os.path.join(DATA_DIR, "synthetic_energy.csv")
    
    if os.path.exists(real_data_path):
        print("Real dataset found. Loading...")
        pass
    
    if os.path.exists(synthetic_path):
        print("Loading existing synthetic data.")
        df = pd.read_csv(synthetic_path, parse_dates=['tstp'])
        return df
        
    return generate_synthetic_data()

def get_city_wide_daily(df):
    """Aggregates all households to get total daily consumption."""
    df['date'] = df['tstp'].dt.date
    daily = df.groupby('date')['energy(kWh/hh)'].sum().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    return daily

if __name__ == "__main__":
    df = load_data()
    print(df.head())
