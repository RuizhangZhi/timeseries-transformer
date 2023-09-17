import os
from dotenv import load_dotenv
import pandas as pd  
from nixtlats import TimeGPT

# Initializing the `TimeGPT` class with a token from the environment variable.
load_dotenv()
MY_SECRET_KEY = os.environ.get('MY_NIXTLA_SECRET_KEY')
timegpt = TimeGPT(token=MY_SECRET_KEY) 
if timegpt.validate_token():
    print("Token validation successful!")
else:
    raise Exception("Token validation failed! Please check go to https://dashboard.nixtla.io/ to get your token.")

# Loading the air passengers dataset from a remote URL as an example
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')

# Forecasting the next 12 horizons using TimeGPT
timegpt_fcst_df = timegpt.forecast(df=df, h=12, time_col='timestamp', target_col='value')

# Plotting the original data combined with the forecasted data.
pd.concat([df, timegpt_fcst_df]).set_index('timestamp').plot()
