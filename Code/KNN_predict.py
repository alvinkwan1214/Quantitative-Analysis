import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf


# Step 1: Load the Data
# Replace 'your_stock_data.csv' with your dataset path
ticker = "AAPL"
stdate = '2013-12-14'
eddate = '2023-12-14'

data = yf.download(ticker,start=stdate,end=eddate)
data = data.reset_index()
print(data)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 2: Preprocess the Data
# Use features like 'Open', 'High', 'Low', 'Close', 'Volume'
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# Create lagged features to use previous days' prices as predictors
for lag in range(1, 6):
    for feature in features:
        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)

# Drop rows with NaN values due to shifting
data.dropna(inplace=True)

# Define X and y
X = data[[col for col in data.columns if 'lag' in col]]
y = data[target]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Apply KNN Algorithm
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = knn.predict(X_test_scaled)

# Step 7: Evaluate the Model
# Plot the results
plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test.values, label='Actual Price', color='b')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='r')
plt.title('Stock Price Prediction using KNN')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f'Root Mean Squared Error: {rmse}')
