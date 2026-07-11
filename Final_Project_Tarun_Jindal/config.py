import os

# Define global configuration variables and paths
DATA_DIR = '/kaggle/input/datasets/jeanmidev/smart-meters-in-london/'

# Output directory for saving generated plots and processed data
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Basic variables for memory limits to prevent RAM crashes
NROWS = 100000
