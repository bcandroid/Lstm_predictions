from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
import pywt
## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

df = pd.read_csv('./household_power_consumption.txt',sep = ';',
                parse_dates={'dt':['Date','Time']},
                infer_datetime_format=True,
                low_memory=False, na_values=['nan','?'],
                index_col='dt')
droping_list_all=[]
for j in range(0, 7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
droping_list_all


for j in range(0,7):
    df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]
        
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
df_resample = df.resample('h').mean()
values = df_resample.values
train_data, temp_data = train_test_split(values, test_size=0.3, shuffle=False)
test_data, validation_data = train_test_split(temp_data, test_size=0.5, shuffle=False)
# Remove '.values' when reshaping
train_data = train_data.reshape(-1, 1)
validation_data = validation_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
validation_data = scaler.transform(validation_data)
def apply_dwt(data, wavelet='db1', level=None):
    if level is None:
        level = int(np.log2(len(data)))
    
    coeffs = pywt.wavedec(data, wavelet, level=min(level, pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet))))
    flattened_coeffs = [c for sublist in coeffs for c in sublist]
    return flattened_coeffs
def create_groups(dataset, window_size_2, window_size_3, timeslice_2, timeslice_3, step, d, p):
    X_data, y_data = [], []
    index = 0
    while index + (timeslice_3 * window_size_3) < len(dataset):
        i = 0
        t2, t3 = [], []
        while i < timeslice_3 * window_size_3:
            current_slice = dataset[index + i:index + i + window_size_3]
            if not np.isnan(current_slice).all():
                t3.append(np.mean(current_slice))
            i += window_size_3

        # Populate t2
        t2 = []
        j = timeslice_3 * window_size_3 - timeslice_2 * window_size_2
        while j < i:
            current_slice = dataset[index + j:index + j + window_size_2]
            if not np.isnan(current_slice).all():
                t2.extend(current_slice)
            j += window_size_2

        # Transpoz işlemi
        t3 = np.array(t3).reshape(-1, 1)
        t2 = np.array(t2).reshape(1, -1)
        t3 = np.transpose(t3)
        t3=apply_dwt(t3)
        t2=apply_dwt(t2)
        m1 = np.mean(t3)
        m2 = np.mean(t2)
        t3 = t3 - m1
        t2 = t2 - m2
        my_ar = np.full((timeslice_1,), m1)
        my_arr = np.full((timeslice_2,), m2)
        my_array = np.concatenate([my_arr, my_ar], axis=0)
        d.append(my_array)
        p.append(m1)
        concatenated_slices = np.concatenate([t2, t3], axis=1)
        X_data.append(concatenated_slices)
        y_data = np.append(y_data, dataset[index + timeslice_3 * window_size_3] - m1)
        index += step
    X_data = np.array(X_data)
    a, b, c = X_data.shape
    array = X_data.reshape(a * b, c)
    return array, np.array(y_data)

window_size_1 = 1
window_size_2 = 25
timeslice_1= 4
timeslice_2 =4
step = 1
xtrm=[]
ytrm=[]
xtem=[]
ytem=[]
X_train, y_train =create_groups(train_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtrm,ytrm)
X_validation, y_validation = create_groups(validation_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,xtem,ytem)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Reshape inputs for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
# Building the LSTM Model
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dropout, Dense
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

l,k=[],[]
X_test, y_test = create_groups(test_data,window_size_1, window_size_2, timeslice_1,timeslice_2,step,l,k)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = y_test.reshape(-1, 1)
predicted_stock_price = model.predict(X_test)
k=np.array(k)
k = k.reshape(-1, 1)
y_test=y_test+k
predicted_stock_price=predicted_stock_price+k
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

y_test = scaler.inverse_transform(y_test)
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