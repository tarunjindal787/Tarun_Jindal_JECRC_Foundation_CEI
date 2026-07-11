from preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- RUNNING EDA ---")

# Ensure matplotlib doesn't try to open windows
plt.switch_backend('Agg')

# 1. Line chart of overall consumption trends over time
print("Generating consumption trend plot...")
# Aggregate daily consumption across all households
daily_total = daily_df.groupby('day')['energy_sum'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(daily_total['day'], daily_total['energy_sum'], color='#FF6B6B')
plt.title('Total Daily Energy Consumption in London (Sample)')
plt.xlabel('Date')
plt.ylabel('Total Energy (kWh)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'consumption_trend.png'))
plt.close()

# 2. Correlation heatmap showing how temperature drops match up with energy spikes
print("Generating temperature vs energy correlation heatmap...")
# Merge daily total with daily weather
daily_weather_energy = pd.merge(daily_total, weather_daily_df, left_on='day', right_on='time', how='inner')

if not daily_weather_energy.empty and 'temperatureMax' in daily_weather_energy.columns:
    corr_cols = ['energy_sum', 'temperatureMax']
    if 'temperatureMin' in daily_weather_energy.columns:
        corr_cols.append('temperatureMin')
        
    corr_matrix = daily_weather_energy[corr_cols].corr()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation: Energy vs Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'temperature_correlation.png'))
    plt.close()

# 3. Distributions comparing energy profiles between different socio-economic ACORN groups
print("Generating ACORN demographic distribution plot...")
# Merge daily data with household acorn mapping
daily_acorn = pd.merge(daily_df, households_df, on='LCLid', how='inner')

if not daily_acorn.empty and 'Acorn_Group' in daily_acorn.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Acorn_Group', y='energy_sum', data=daily_acorn, palette='viridis')
    plt.title('Energy Consumption Distribution by ACORN Group')
    plt.xlabel('ACORN Group')
    plt.ylabel('Daily Energy (kWh)')
    plt.xticks(rotation=45)
    # Limit y-axis to remove extreme outliers for better visualization
    plt.ylim(0, daily_acorn['energy_sum'].quantile(0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'acorn_distribution.png'))
    plt.close()

print("EDA complete. Visualizations saved to output directory.")
