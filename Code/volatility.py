import numpy as np
import pandas as pd
import yfinance as yf

def calculate_volatility(stock_symbol, start_date, end_date):
    # Download historical stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()

    # Calculate the annualized volatility
    volatility = stock_data['Daily Return'].std() * np.sqrt(252)  # 252 trading days in a year

    return volatility

# Example usage
stock_symbol = 'AAPL'  # Apple's stock symbol
start_date = '2023-01-01'
end_date = '2023-12-31'

volatility = calculate_volatility(stock_symbol, start_date, end_date)
print(f"Annualized Volatility for {stock_symbol}: {volatility:.4f}")
