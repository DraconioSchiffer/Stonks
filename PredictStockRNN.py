import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
import math

def plot_series(time, series, start=0, end=None, format="-", label = ""):
  plt.plot(time[start:end], series[start:end], format, label = label)
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.grid(True)  

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)    
    ds = ds.window(window_size, shift=1, drop_remainder=True)    
    ds = ds.flat_map(lambda w: w.batch(window_size))    
    ds = ds.batch(32).prefetch(1)        
    forecast = model.predict(ds)    
    return forecast
  
cst = 'AAPL'
st = ['AAPL','GOOGL','FB','AMZN','MSFT','T','INTC','WMT','GM','JPM']
print("Available stocks are:")
for i in range(len(st)):
  print(i,st[i])
choice = int(input("Enter sl no of stock to chose:\n"))

stock = pd.read_csv('data/' + st[choice] + '.csv')
time_stamp = stock['Date']
series = stock['Adj Close']
series = np.array(series)

time = []
for i in range(len(series)):
  time.append(i)
plot_series(time,series)
plt.title(st[choice] + " stock from " + time_stamp[0] + " to " + time_stamp[len(time_stamp)-1])
plt.show()
trt = input("Press Enter to continue")

dividing_factor = math.floor(max(series)/300)
if dividing_factor == 0:
  dividing_factor = 1

series /= dividing_factor

split_time = 1300
time_train = time[:split_time]   #WE ARE TRAINING THE WHOLE SERIES THIS IS JUST FOR VISUALIZATION PURPOSE
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 10



#This is just to pick out the optimum learning rate
tf.keras.backend.clear_session()
train_set = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])


#Seeing lr to get optimum lr
plt.semilogx(history.history["lr"], history.history["loss"])
plt.title("Choose learning rate at a stable point")
plt.show()


learning_rate = float(input("Enter the learning rate in exponential form: (eg: 8e-7 = 8 * 10^-7):\n"))


#This is now the actual model
tf.keras.backend.clear_session()
train_set = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=150)



rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)  #Adds a new dimension. The ... is called an ellipses object
b = rnn_forecast[:,-1,0]
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
'''
plt.figure(figsize=(10, 6))
plot_series(time,series)
plot_series(time[window_size-1:],b)
#plot_series(time_valid, x_valid)
plot_series(time_valid,rnn_forecast)
#plt.axis([1470,1510,50,60])
#plt.axis([0,35,0,100])
#plot_series(newtime, prediction)
plt.show()
'''

#predict number of days
print("\n(Note: Increasing number of days increases error)")
no_days = int(input("Enter no of days in the future to predict: "))
f = copy.deepcopy(series[len(series)-window_size:len(series)])
maxtime = time[len(time)-1] + 1
newtime = [len(series)-1]
prediction = [b[len(b)-1]]
for i in range(no_days):    #Here range is number of days 
  a = model_forecast(model, f[...,np.newaxis], window_size)  
  ele = a[:,-1,0]
  newele = np.ndarray.tolist(ele)[0]
  prediction.append(newele)
  newtime.append(maxtime)
  maxtime+=1
  f = f[1:]  
  f = np.append(f,ele)  
print(newtime)
print(prediction)

plt.figure(figsize=(10, 6))
#------------------
series *= dividing_factor
b *= dividing_factor
prediction = np.array(prediction)
prediction *= dividing_factor
#-----------------
plot_series(time,series, label = "Real stock data")
plot_series(time[window_size-1:],b, label = "Predicted stock data")
#plot_series(time_valid, x_valid)
#plot_series(time_valid,rnn_forecast)
#plt.axis([1470,1520,50,60])
#plt.axis([0,35,0,100])
plot_series(newtime, prediction, label = "Future prediction")
plt.legend(loc = "upper left")
plt.title("Final prediction")
plt.show()
