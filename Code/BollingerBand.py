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

# Step 2: Calculate Bollinger Bands
def bollinger_bands(data):
    window = 20
    data['Middle_Band'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band'] = data['Middle_Band'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower_Band'] = data['Middle_Band'] - 2 * data['Close'].rolling(window=window).std()
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'] = np.where(data['Close'] < data['Lower_Band'], 1, 0)
    data['Signal'] = np.where(data['Close'] > data['Upper_Band'], -1, data['Signal'])
    data['Position'] = data['Signal'].diff()
    
    return data

# Step 3: Implement Backtesting
def backtesting(data, initial_capital):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)

    positions['Stock'] = data['Signal'] * 100  # Number of stocks to trade
    portfolio['Positions'] = positions.multiply(data['Close'], axis=0)

    portfolio['Cash'] = initial_capital - (positions.diff().multiply(data['Close'], axis=0)).cumsum().fillna(0)
    portfolio['Total'] = portfolio['Positions'] + portfolio['Cash']
    return data, portfolio

# Step 4: Evaluate the Strategy
def plot_graphs(data, portfolio):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Upper_Band'], label='Upper Band')
    plt.plot(data['Middle_Band'], label='Middle Band')
    plt.plot(data['Lower_Band'], label='Lower Band')
    plt.plot(data.loc[data['Position'] == 1.0].index, data['Close'][data['Position'] == 1.0], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data.loc[data['Position'] == -1.0].index, data['Close'][data['Position'] == -1.0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Bollinger Bands Strategy')
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
    df_signal = bollinger_bands(df)
    final_result, portfolio = backtesting(df_signal, initial_capital)
    plot_graphs(final_result, portfolio)
    investment_performance(portfolio)
