import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Step 1: Download Historical Data
ticker = 'AAPL'
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
closing_prices = data['Adj Close']

# Step 2: Calculate Daily Logarithmic Returns
log_returns = np.log(closing_prices / closing_prices.shift(1)).dropna()

# Step 3: Plot the Data

# Plot the Closing Prices
plt.figure(figsize=(14, 7))
plt.plot(closing_prices, label='Adjusted Close Price')
plt.title(f'{ticker} Adjusted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the Log Returns
plt.figure(figsize=(14, 7))
plt.plot(log_returns, label='Logarithmic Returns')
plt.title(f'{ticker} Daily Logarithmic Returns')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# Step 4: Stationarity Check using ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(f'{label}: {value}')
    
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

# Perform ADF test on Closing Prices
adf_test(closing_prices)

# Perform ADF test on Log Returns
adf_test(log_returns)
