import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def get_stock_info(ticker, stdate, eddate):
    data = yf.download(ticker,start=stdate,end=eddate)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data


def sm_signal(data):
    short_window = 40  # Short-term moving average window
    long_window = 100  # Long-term moving average window

    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Step 3: Generate Buy/Sell Signals
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data


def backtesting(data, initial_capital):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)

    positions['Stock'] = data['Signal'] * 100  # Number of stocks to trade 
    portfolio['Positions'] = positions.multiply(data['Close'], axis=0) #pay

    portfolio['Cash'] = initial_capital - (positions.diff().multiply(data['Close'], axis=0)).cumsum()
    portfolio['Total'] = portfolio['Positions'] + portfolio['Cash']
    print(df["Position"].astype(bool).sum(axis=0))
    return data, portfolio

def plot_graphs(data, portfolio):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Short_MA'], label='Short-term MA (40)')
    plt.plot(data['Long_MA'], label='Long-term MA (100)')
    plt.plot(data.loc[data['Position'] == 1.0].index, data['Short_MA'][data['Position'] == 1.0], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data.loc[data['Position'] == -1.0].index, data['Short_MA'][data['Position'] == -1.0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot the portfolio value over time
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio['Total'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


    # Print final portfolio value
    plt.figure(figsize=(14, 7))
    plt.plot(data['Position'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio['Positions'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
    return

def investment_performance(final_result, portfolio):
    print(f'Final Portfolio Value: ${portfolio["Total"].iloc[-1]:.2f}')
    earning = portfolio["Total"].iloc[-1] - portfolio["Total"].iloc[1]
    print(f'Final Portfolio Value: ${earning:.2f}')

if __name__ == "__main__":
    ticker = "AAPL" 
    stdate = '2020-12-14'
    eddate = '2023-12-14'
    initial_capital = 10000.0  # Starting capital
    df = get_stock_info(ticker, stdate, eddate)
    df_signal = sm_signal(df)
    final_result, portfolio = backtesting(df_signal, initial_capital)
    plot_graphs(final_result, portfolio)
    investment_performance(final_result, portfolio)
    
