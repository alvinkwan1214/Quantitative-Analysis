import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler



class stock():
    def __init__(self, stock_name, start_date, end_date):
        self.stock_data = yf.download(stock_name, start=start_date, end=end_date)
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        return self.stock_data

    def __str__(self):
        return f"Stock Data for {self.stock_name} from {self.start_date} to {self.end_date}"
    
    def show_data(self, head=True):
        if head:
            print(self.stock_data.head())
        else:
            print(self.stock_data)

    def plot_data(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Adj Close'], label='Adjusted Close Price')
        plt.title(f'{self.stock_name} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    def adj_close(self):
        adj_close = self.stock_data['Adj Close']
        #adj_close['Adj Close'] = self.stock_name
        return adj_close

    

stock_list = pd.read_csv('bats_symbols.csv').Name.to_list()
test_list = stock_list[:100]

for i in range(len(test_list)):
    stock_test = stock(test_list[i], '2023-01-01', '2023-12-31')
    if i == 0:
        df = stock_test.adj_close()
    else:
        df = pd.concat([df, stock_test.adj_close()], axis=1)

missing = df.isnull().sum() / len(df) 
missing

features_to_drop = missing[missing > 0.3].index
X_train = df.drop(columns=features_to_drop)
X_test = df.drop(columns=features_to_drop)








