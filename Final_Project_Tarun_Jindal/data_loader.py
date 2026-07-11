import pandas as pd
import numpy as np
import os
import config

print("--- RUNNING DATA LOADER ---")

# Ingest raw dataset files directly using sequential pandas commands
households_file = os.path.join(config.DATA_DIR, 'informations_households.csv')
bank_holidays_file = os.path.join(config.DATA_DIR, 'uk_bank_holidays.csv')
weather_daily_file = os.path.join(config.DATA_DIR, 'weather_daily_darksky.csv')
weather_hourly_file = os.path.join(config.DATA_DIR, 'weather_hourly_darksky.csv')
acorn_file = os.path.join(config.DATA_DIR, 'acorn_details.csv')
daily_dataset_file = os.path.join(config.DATA_DIR, 'daily_dataset.csv')

# Use dummy data fallback for testing locally if Kaggle paths don't exist
if not os.path.exists(config.DATA_DIR):
    print("WARNING: Kaggle paths not found. Creating dummy data arrays for pipeline execution.")
    # Create tiny dummy DataFrames to allow the pipeline to run successfully without crashing
    households_df = pd.DataFrame({'LCLid': ['MAC00001', 'MAC00002'], 'Acorn': ['ACORN-A', 'ACORN-B']})
    bank_holidays_df = pd.DataFrame({'Bank holidays': ['2012-01-02']})
    weather_daily_df = pd.DataFrame({'time': ['2012-01-01', '2012-01-02'], 'temperatureMax': [10.5, 9.2]})
    weather_hourly_df = pd.DataFrame({'time': ['2012-01-01 00:00:00'], 'temperature': [5.5]})
    acorn_df = pd.DataFrame({'ACORN-A': [1], 'ACORN-B': [2]})
    
    dates = pd.date_range('2012-01-01', periods=20, freq='D')
    daily_df = pd.DataFrame({
        'LCLid': ['MAC00001'] * 20 + ['MAC00002'] * 20,
        'day': list(dates) + list(dates),
        'energy_sum': np.random.uniform(5, 15, 40).tolist()
    })
else:
    # Load from Kaggle path
    print("Loading informations_households.csv")
    households_df = pd.read_csv(households_file)
    
    print("Loading uk_bank_holidays.csv")
    bank_holidays_df = pd.read_csv(bank_holidays_file)
    
    print("Loading weather_daily_darksky.csv")
    weather_daily_df = pd.read_csv(weather_daily_file)
    
    print("Loading weather_hourly_darksky.csv")
    weather_hourly_df = pd.read_csv(weather_hourly_file)
    
    print("Loading acorn_details.csv")
    # Load acorn explicitly using encoding='latin1' to prevent encoding bugs
    acorn_df = pd.read_csv(acorn_file, encoding='latin1')
    
    print(f"Loading daily_dataset.csv (Limit: {config.NROWS} rows)")
    # Load the main consumption files utilizing the nrows=NROWS configuration
    daily_df = pd.read_csv(daily_dataset_file, nrows=config.NROWS)

print("Data loading complete.")
