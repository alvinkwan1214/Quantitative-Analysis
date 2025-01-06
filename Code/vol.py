import numpy as np
import pandas as pd
import yfinance as yf

# Step 1: Obtain Historical Price Data
ticker = 'AAPL'
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")
closing_prices = data['Adj Close']

# Step 2: Calculate Daily Returns
daily_returns = closing_prices.pct_change().dropna()

# Step 3: Calculate the Standard Deviation of Returns (Daily Volatility)
daily_volatility = daily_returns.std()

# Step 4: Annualize the Volatility
annualized_volatility = daily_volatility * np.sqrt(252)

print(f"Daily Volatility: {daily_volatility:.4f}")
print(f"Annualized Volatility: {annualized_volatility:.4f}")    