from data_loader import *
import pandas as pd

print("--- RUNNING PREPROCESSING ---")

# Convert all target date and time columns into proper pandas datetime objects
print("Formatting date columns...")
if 'day' in daily_df.columns:
    daily_df['day'] = pd.to_datetime(daily_df['day'])

if 'time' in weather_daily_df.columns:
    weather_daily_df['time'] = pd.to_datetime(weather_daily_df['time'])

if 'time' in weather_hourly_df.columns:
    weather_hourly_df['time'] = pd.to_datetime(weather_hourly_df['time'])
    
if 'Bank holidays' in bank_holidays_df.columns:
    bank_holidays_df['Bank holidays'] = pd.to_datetime(bank_holidays_df['Bank holidays'], format='%Y-%m-%d', errors='coerce')

# Clean string formatting (stripping out the 'ACORN-' prefix text)
print("Cleaning ACORN classifications...")
if 'Acorn' in households_df.columns:
    # Handle cases where value might not have ACORN- prefix smoothly
    households_df['Acorn_Group'] = households_df['Acorn'].astype(str).str.replace('ACORN-', '', regex=False)

# Fill missing environmental weather rows sequentially using forward-fill methods
print("Imputing missing weather values...")
# Sort by time to ensure forward-fill works correctly chronologically
weather_daily_df = weather_daily_df.sort_values('time')
weather_daily_df = weather_daily_df.ffill()

weather_hourly_df = weather_hourly_df.sort_values('time')
weather_hourly_df = weather_hourly_df.ffill()

print("Preprocessing complete.")
