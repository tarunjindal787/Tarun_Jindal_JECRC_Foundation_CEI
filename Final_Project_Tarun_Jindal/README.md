# Smart Meters Energy Analytics Project

## What this project is
This is a data science project built to analyze electricity consumption across London households. The goal is to take raw smart meter logs, combine them with local weather conditions and calendar holidays, and use machine learning to understand how people use power and predict future energy loads.

## Project Objective
The main goal of this system is to study daily electrical consumption patterns. By matching energy usage to household demographics and weather metrics, the project helps find ways to optimize the energy grid, understand different neighborhood habits, and provide insights that can help tackle climate change through smarter energy management.

## Repository Structure
The project is split into separate, sequential files with no complex custom functions so that the entire pipeline is flat and easy to follow line-by-line:

*   config.py: Holds global setup variables, data directory paths, and row limits to protect notebook memory.
*   data_loader.py: Safely loads the raw CSV files using pandas and limits rows on the massive consumption datasets to prevent RAM crashes.
*   preprocessing.py: Formats date columns into datetime objects, cleans categorical text, and fixes missing weather values.
*   eda.py: Generates and saves charts showing consumption trends over time, temperature correlations, and demographic breakdowns.
*   feature_engineering.py: Merges all dataframes together on timestamps and creates lag variables and rolling averages.
*   train.py: Splitting the final data chronologically and training an XGBoost/LightGBM regressor model to forecast future electrical loads.
*   main.py: The single entry point script used to run files 1 through 6 in a continuous order.
