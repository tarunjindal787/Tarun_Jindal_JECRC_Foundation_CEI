# The single entry point script used to run files 1 through 6 in a continuous order.
# Because the scripts are written sequentially without functions, importing them 
# executes them from top to bottom.

print("Starting Smart Meters Energy Analytics Pipeline...")

# This will trigger the cascading execution of:
# config -> data_loader -> preprocessing -> eda -> feature_engineering -> train
import train

print("Pipeline finished.")
