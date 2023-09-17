# Importing necessary modules.
import pandas as pd  
from nixtlats import TimeGPT  # Importing the TimeGPT class from nixtlats library.

# Initializing the `TimeGPT` class with a token from the environment variable.
timegpt = TimeGPT(token='YOUR-TIMEGPT_TOKEN') #https://dashboard.nixtla.io

# Check if the token provided is valid.
if timegpt.validate_token():
    print("Token validation successful!")  # Token is valid.
else:
    # Raise an exception if token validation fails.
    raise Exception("Token validation failed! Please check go to https://dashboard.nixtla.io/ to get your token.")

# Loading the air passengers dataset from a remote URL as an example
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')

# Forecasting the next 12 horizons using TimeGPT
timegpt_fcst_df = timegpt.forecast(df=df, h=12, time_col='timestamp', target_col='value')

# Plotting the original data combined with the forecasted data.
pd.concat([df, timegpt_fcst_df]).set_index('timestamp').plot()
