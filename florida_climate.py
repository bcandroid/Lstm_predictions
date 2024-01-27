import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import pywt
import random
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
florida = pd.read_csv('./florida_file_.csv')
# Veriyi işle
florida["Date"] = pd.to_datetime(florida["Date"])
florida = florida[["Date", "Avg_Temp"]]
florida = florida.fillna(florida.bfill())
florida.columns = ['Date', 'Avg_Temp']

# Veriyi ölçeklendir
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(florida['Avg_Temp'].values.reshape(-1, 1))

# Veriyi train, test ve validasyon olarak ayır
training_data, temp_veri = train_test_split(scaled_data, test_size=0.4, random_state=42)
test_data, validation_data = train_test_split(temp_veri, test_size=0.5, random_state=42)


scaled_data_df = pd.DataFrame(scaled_data, columns=['Avg_Temp'])
training_data, temp_veri = train_test_split(scaled_data_df, test_size=0.4, random_state=42)
test_data, validation_data = train_test_split(temp_veri, test_size=0.5, random_state=42)

# Sonuçları kontrol et
print("Train Verisi Boyutu:", len(training_data))
print("Test Verisi Boyutu:", len(test_data))
print("Validation Verisi Boyutu:", len(validation_data))

# Use 'Avg_Temp' column as the stock price for training
training_set = training_data['Avg_Temp'].values.reshape(-1, 1)
validation_set = validation_data['Avg_Temp'].values.reshape(-1, 1)
test_set = test_data['Avg_Temp'].values.reshape(-1, 1)

# Feature scaling using MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.transform(validation_set)
test_set_scaled = sc.transform(test_set)

def create_sequences(data, seq_length=255):
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

X_test, y_test = create_sequences(test_set_scaled)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = y_test.reshape(-1, 1)
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_test = sc.inverse_transform(y_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
# Evaluate the model
mse = mean_squared_error(y_test, predicted_stock_price)

mae = mean_absolute_error(y_test, predicted_stock_price)
r2 = r2_score(y_test,predicted_stock_price)
# Assuming y_pred and y_test are NumPy arrays
# Note: For MAPE, make sure y_test does not contain zeros to avoid division by zero.
# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
# MAPE
mape = np.mean(np.abs((y_test -predicted_stock_price) / y_test)) * 100
print('RMSE:', rmse)
print('MAPE:', mape)
print('MSE: ', mse)
print('MAE: ', mae)
print('R-squared: ', r2)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted_stock_price, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
