from eda import *
import pandas as pd

print("--- RUNNING FEATURE ENGINEERING ---")

# Merge household rows with demographic descriptions from the acorn dataframe
print("Merging datasets...")
merged_df = pd.merge(daily_df, households_df, on='LCLid', how='left')

# Merge the meter data with daily weather dataframes using the timestamp columns
# We use 'day' from daily_df and 'time' from weather_daily_df
merged_df = pd.merge(merged_df, weather_daily_df, left_on='day', right_on='time', how='left')

# Drop redundant time column after merge
if 'time' in merged_df.columns:
    merged_df = merged_df.drop('time', axis=1)

# Ensure day is a datetime object
merged_df['day'] = pd.to_datetime(merged_df['day'])

# Sort chronologically for each household to compute accurate lags
merged_df = merged_df.sort_values(['LCLid', 'day'])

# Compute new shift/lag columns, rolling averages sequentially
print("Computing lags and rolling averages...")
merged_df['energy_lag_1'] = merged_df.groupby('LCLid')['energy_sum'].shift(1)
merged_df['energy_lag_7'] = merged_df.groupby('LCLid')['energy_sum'].shift(7)

merged_df['energy_rolling_7'] = merged_df.groupby('LCLid')['energy_sum'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Calendar flags (like weekend indicators) sequentially
print("Extracting calendar features...")
merged_df['day_of_week'] = merged_df['day'].dt.dayofweek
merged_df['month'] = merged_df['day'].dt.month
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Merge with bank holidays
bank_holidays_df['is_holiday'] = 1
merged_df = pd.merge(merged_df, bank_holidays_df, left_on='day', right_on='Bank holidays', how='left')
merged_df['is_holiday'] = merged_df['is_holiday'].fillna(0).astype(int)
if 'Bank holidays' in merged_df.columns:
    merged_df = merged_df.drop('Bank holidays', axis=1)

# Drop NA rows caused by lag shifts
print("Dropping NA rows created by lag features...")
merged_df = merged_df.dropna(subset=['energy_lag_1', 'energy_lag_7'])

print("Feature engineering complete. Merged shape:", merged_df.shape)
