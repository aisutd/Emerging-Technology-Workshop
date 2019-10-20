import os
import pprint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


pp = pprint.PrettyPrinter()
datafile = 'bitcoin.csv'
prices = pd.read_csv(datafile, index_col=0)
prices = prices[::-1]

print(prices)

def naive_preprocess(dataframe):
  new_dataframe = dataframe.drop(columns=['Volume'])
  market_cap_column = new_dataframe['Market Cap']
  new_dataframe['Market Cap'] = [int(market_cap.replace(',', '')) for market_cap in market_cap_column]
  return new_dataframe

prices_dataframe = naive_preprocess(prices)
print(prices_dataframe)

# plt.plot(prices_dataframe['Open'])
# plt.show()

prices_scaler = MinMaxScaler(feature_range=(0, 1))
market_cap_scaler = MinMaxScaler(feature_range=(0, 1))

values = prices_dataframe.values
values[:, :-1] = prices_scaler.fit_transform(values[:, :-1].reshape(-1, 1)).reshape(values[:, :-1].shape)
values[:, -1] = market_cap_scaler.fit_transform(values[:, -1].reshape(-1, 1)).flatten()
x = values[:-1]
y = values[1:, 0]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
train_y = train_y.reshape(train_y.shape[0], 1)
:set nonumberimport os
import pprint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


pp = pprint.PrettyPrinter()
datafile = 'bitcoin.csv'
prices = pd.read_csv(datafile, index_col=0)
prices = prices[::-1]

print(prices)

def naive_preprocess(dataframe):
  new_dataframe = dataframe.drop(columns=['Volume'])
  market_cap_column = new_dataframe['Market Cap']
  new_dataframe['Market Cap'] = [int(market_cap.replace(',', '')) for market_cap in market_cap_column]
  return new_dataframe

prices_dataframe = naive_preprocess(prices)
print(prices_dataframe)

# plt.plot(prices_dataframe['Open'])
# plt.show()

prices_scaler = MinMaxScaler(feature_range=(0, 1))
market_cap_scaler = MinMaxScaler(feature_range=(0, 1))

values = prices_dataframe.values
values[:, :-1] = prices_scaler.fit_transform(values[:, :-1].reshape(-1, 1)).reshape(values[:, :-1].shape)
values[:, -1] = market_cap_scaler.fit_transform(values[:, -1].reshape(-1, 1)).flatten()
x = values[:-1]
y = values[1:, 0]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
train_y = train_y.reshape(train_y.shape[0], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
test_y = test_y.reshape(test_y.shape[0], 1)

model = tf.keras.Sequential([
  tf.keras.layers.LSTM(16, input_shape=train_x.shape[1:], activation='tanh'),
  tf.keras.layers.Dense(1)])

print(model.summary())

model.compile('rmsprop', loss='mse')

model.fit(train_x, train_y, epochs=20, batch_size=128)

preds = model.predict(test_x)
print(preds.shape)


print(prices_scaler.inverse_transform(preds)[:10])
print(prices_scaler.inverse_transform(test_y)[:10])


