import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
import pywt
import random
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
# CSV dosyasını oku
df = pd.read_csv("KwhConsumptionBlower78_1.csv")

# 'Consumption' sütununu tut
veri = df['Consumption']

# 'TxnDate' sütununu tut
txn_date_column = df['TxnDate']

# Veriyi train, test ve validasyon olarak ayır
training_data, temp_veri = train_test_split(veri.values, test_size=0.4, random_state=42)
test_data, validation_data = train_test_split(temp_veri, test_size=0.5, random_state=42)

# Sonuçları kontrol et
print("Train Verisi Boyutu:", len(training_data))
print("Test Verisi Boyutu:", len(test_data))
print("Validation Verisi Boyutu:", len(validation_data))

# Use 'Adj Close' prices as the stock price for training
training_set = training_data.reshape(-1, 1)
validation_set = validation_data.reshape(-1, 1)
test_set = test_data.reshape(-1, 1)

# Feature scaling using MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.transform(validation_set)
test_set_scaled = sc.transform(test_set)

def create_sequences(data, seq_length=64):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(validation_set_scaled)

# Reshape inputs for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
# Building the LSTM Model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model and store history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))

