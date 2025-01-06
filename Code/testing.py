import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Load the Data
def get_stock_info(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data




# Step 2: Calculate Moving Averages
def ma_signal(data):
    short_window = 40
    long_window = 100

    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    data['MA_Signal'] = 0
    data['MA_Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, -1)
    data['MA_Position'] = data['MA_Signal'].diff()

    return data

# Step 3: Calculate RSI and Combine Signals
def combined_signal(data):
    # MA
    short_window = 40
    long_window = 100

    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # RSI
    window = 14
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    data['RSI'] = RSI

    # Generate signals
    data['Signal'] = 0
    data['Signal'] = np.where(
        (data['RSI'] < 30) & (data['Short_MA'] > data['Long_MA']), 1,
        np.where((data['RSI'] > 70) & (data['Short_MA'] < data['Long_MA']), -1, 0)
    )
    data['Position'] = data['Signal'].diff()
    print(df["Position"].astype(bool).sum(axis=0))
    return data

# Step 4: Implement Backtesting
def backtesting(data, initial_capital):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)

    positions['Stock'] = data['Signal'] * 100  # Number of stocks to trade
    portfolio['Positions'] = positions.multiply(data['Close'], axis=0)

    portfolio['Cash'] = initial_capital - (positions.diff().multiply(data['Close'], axis=0)).cumsum().fillna(0)
    portfolio['Total'] = portfolio['Positions'] + portfolio['Cash']

    return data, portfolio

# Step 5: Evaluate the Strategy
def plot_graphs(data, portfolio):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Short_MA'], label='Short-term MA (40)')
    plt.plot(data['Long_MA'], label='Long-term MA (100)')
    plt.plot(data.loc[data['Position'] == 1.0].index, data['Short_MA'][data['Position'] == 1.0], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data.loc[data['Position'] == -1.0].index, data['Short_MA'][data['Position'] == -1.0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Combined MA and RSI Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio['Total'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

def investment_performance(portfolio):
    print(f'Final Portfolio Value: ${portfolio["Total"].iloc[-1]:.2f}')
    earning = portfolio["Total"].iloc[-1] - portfolio["Total"].iloc[0]
    print(f'Net Earning: ${earning:.2f}')

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = '2020-12-14'
    end_date = '2023-12-14'
    initial_capital = 10000.0  # Starting capital

    df = get_stock_info(ticker, start_date, end_date)
    df_signal = combined_signal(df)
    final_result, portfolio = backtesting(df_signal, initial_capital)
    plot_graphs(final_result, portfolio)
    investment_performance(portfolio)
