# Final Project: London Smart Meter Energy Analytics
**Author:** Tarun Jindal

Welcome to my final project for the JECRC Foundation CEI program! 

This repository contains the source code for my end-to-end Energy Analytics system. I built this dashboard to analyze smart meter data from London households and apply machine learning to forecast energy consumption and detect anomalies.

## Project Motivation
The core objectives behind this initiative are:
1. **Grid Modernization**: To help the British government and energy suppliers understand complex energy consumption patterns in order to upgrade an aging electrical grid.
2. **Rollout Preparation**: To gather baseline data and trial-run insights ahead of massive, nationwide smart meter rollouts.
3. **Climate & Sustainability Action**: To analyze how micro-habits, demographic backgrounds, and weather conditions impact energy demand, allowing suppliers to build localized energy-saving strategies and tackle climate change.

## Tech Stack
- **Python**: Core programming language.
- **Pandas & NumPy**: Data processing and aggregation.
- **Scikit-Learn**: Machine learning models (`RandomForestRegressor` for forecasting, `IsolationForest` for pattern analysis).
- **Streamlit**: Web dashboard framework.
- **Plotly**: Interactive data visualizations.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tarunjindal787/Tarun_Jindal_JECRC_Foundation_CEI.git
   cd Tarun_Jindal_JECRC_Foundation_CEI/Final_Project_Tarun_Jindal
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```
   *(Note: The first time you run the app, it will take a few seconds to run the data pipeline and train the models locally.)*

## Deployment
This project is configured to be easily deployed on Streamlit Community Cloud. Simply connect this GitHub repository and select `Final_Project_Tarun_Jindal/app.py` as the main entry point.

---
*Created as part of the JECRC Foundation CEI initiative.*
