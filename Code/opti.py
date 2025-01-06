import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Step 1: Data Collection
# Download stock data from Yahoo Finance
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']

# Step 2: Calculate Returns
returns = data.pct_change().dropna()

# Step 3: Calculate Mean and Covariance
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Step 4: Portfolio Optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

def maximize_sharpe_ratio(mean_returns, cov_matrix, num_assets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Optimize Portfolio
num_assets = len(tickers)
optimal_portfolio = maximize_sharpe_ratio(mean_returns, cov_matrix, num_assets)

# Step 5: Optimal Portfolio Weights
optimal_weights = optimal_portfolio['x']
print(f"Optimal Portfolio Weights: {optimal_weights}")

# Step 6: Portfolio Performance
opt_returns, opt_std_dev = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe_ratio = -optimal_portfolio['fun']

print(f"Expected Annual Return: {opt_returns * 252:.2f}")
print(f"Annual Volatility (Standard Deviation): {opt_std_dev * np.sqrt(252):.2f}")
print(f"Sharpe Ratio: {opt_sharpe_ratio:.2f}")

# Step 7: Plotting Efficient Frontier
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def efficient_frontier(mean_returns, cov_matrix, num_assets, returns_range):
    bounds = tuple((0, 1) for _ in range(num_assets))
    efficient_portfolios = []
    for ret in returns_range:
        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - ret},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(portfolio_volatility, num_assets * [1. / num_assets,], args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(result['fun'])
    return efficient_portfolios

# Generate Efficient Frontier
returns_range = np.linspace(mean_returns.min(), mean_returns.max(), 100)
efficient_volatility = efficient_frontier(mean_returns, cov_matrix, num_assets, returns_range)

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(opt_std_dev, opt_returns, marker='*', color='r', s=200, label='Optimal Portfolio')
plt.plot(efficient_volatility, returns_range, label='Efficient Frontier')
plt.xlabel('Annual Volatility (Standard Deviation)')
plt.ylabel('Expected Annual Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()
