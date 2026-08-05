import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# Load Wine Dataset
# -----------------------------
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Use Alcohol column as the time series values
df = df[['alcohol']]
df.rename(columns={'alcohol': 'Value'}, inplace=True)

# Create a Date column
df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')


df.loc[5, 'Value'] = np.nan
df.loc[20, 'Value'] = np.nan

print("Missing Values Before:")
print(df.isnull().sum())

df['Value'] = df['Value'].fillna(df['Value'].mean())

print("\nMissing Values After:")
print(df.isnull().sum())


# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as Index
df.set_index('Date', inplace=True)

print(df.head())

# Plot Time Series
plt.figure(figsize=(12,5))
plt.plot(df['Value'])
plt.title("Wine Dataset Time Series")
plt.xlabel("Date")
plt.ylabel("Alcohol")
plt.grid(True)
plt.show()

# Trend Analysis
df['Trend'] = df['Value'].rolling(window=7).mean()

plt.figure(figsize=(12,5))
plt.plot(df['Value'], label='Original')
plt.plot(df['Trend'], color='red', label='Trend')
plt.legend()
plt.show()

# Seasonality
result = seasonal_decompose(df['Value'], model='additive', period=7)

result.plot()
plt.show()

# Lag Features
df['Lag1'] = df['Value'].shift(1)
df['Lag2'] = df['Value'].shift(2)
df['Lag3'] = df['Value'].shift(3)

# Moving Average
df['Moving_Average'] = df['Value'].rolling(window=7).mean()

plt.figure(figsize=(12,5))
plt.plot(df['Value'], label='Original')
plt.plot(df['Moving_Average'], label='Moving Average')
plt.legend()
plt.show()

# Remove NaN values
df.dropna(inplace=True)

# Train-Test Split
train_size = int(len(df) * 0.8)

train = df['Value'][:train_size]
test = df['Value'][train_size:]

# Train ARIMA
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

print(model_fit.summary())


# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluation
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("\nMAE :", mae)
print("RMSE:", rmse)


# Actual vs Forecast
comparison = pd.DataFrame({
    'Actual': test,
    'Forecast': forecast
})

print("\nComparison:")
print(comparison.head())

plt.figure(figsize=(12,5))
plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()

# Future Forecast (30 Days)
future = model_fit.forecast(steps=30)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq='D'
)

future_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': future
})

print("\nFuture Forecast:")
print(future_df)

plt.figure(figsize=(12,5))
plt.plot(df['Value'], label='Original')
plt.plot(future_dates, future, color='red', label='Future Forecast')
plt.legend()
plt.show()