#Trend
# Seasonality
# Cyclinder Patterns
import pandas as pd 
import matplotlib.pyplot as plt

data = {
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Sales' : [100, 120, 130, 125, 150, 170, 160, 180, 200, 210]
    
}

df = pd.DataFrame(data)
#print(df)

plt.plot(df['Date'], df['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Daily Sales over Time')
#plt.show()


#Moving Averages

# ex = if a shop has daily sales there flactuation blw sales every day.

df['Moving_Avg'] = df['Sales'].rolling(window=3).mean()
#print(df)

plt.plot(df['Date'], df['Sales'], label = 'Original')
plt.plot(df['Date'], df['Moving_Avg'], label='Moving Average')
plt.legend()
plt.show()