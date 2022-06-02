from operator import mod
from statistics import mode
from matplotlib.style import use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import json

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.utils.vis_utils import plot_model

# Loading data

with open('./config.json') as f:
	config = json.load(f)

use_mock_data = config["use_mock_data"]
company = config["company"]
epochs = config["epochs"]
batch_size = config["batch_size"]
timestep = config["timestep"]

test_start_date = config["test_start_date"].split(".")
test_end_date = config["test_end_date"].split(".")

train_start_day = config["train_start_date"].split(".")

start_date = dt.datetime(int(test_start_date[2]), int(test_start_date[1]), int(test_start_date[0]))
end_date = dt.datetime(int(test_end_date[2]), int(test_end_date[1]), int(test_end_date[0]))

data = pdr.DataReader(company, 'yahoo', start=start_date, end=end_date)

# Preparing data
# Fitting data into 0 to 1 range
# To produce the best-optimized results with the models, we are required to scale the data.
scaler = MinMaxScaler(feature_range=(0,1))

# reshape(-1, 1) turning columns to rows (only one column remains)
if not use_mock_data:
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))
else:
    """ MOCK DATA """
    mock_data = []
    with open("step-data.txt", "r") as f:
        line = f.readline()

    for value in line.split(" "):
        if value:
            mock_data.append(float(value))

    mock_data_array = np.array(mock_data)
    scaled_mock = scaler.fit_transform(mock_data_array.reshape(-1, 1))
    """ --------- """

# x_train represents input data
x_train = []

# y_train represents target data 
y_train = []

def useTestData():
    for i in range(timestep, len(scaled_data)):
        # appending scaled data to x_train from 
        x_train.append(scaled_data[i-timestep:i, 0])
        y_train.append(scaled_data[i, 0])

def useMockData():
    for i in range(timestep, len(scaled_mock)):
        # appending scaled data to x_train from 
        x_train.append(scaled_mock[i-timestep:i, 0])
        y_train.append(scaled_mock[i, 0])

if use_mock_data:
    useMockData()
else:
    useTestData()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


""" BUILDING THE MODEL """

# Adding layers to our seqeuntial model with dropout between each layer
# Dropout is easily implemented by randomly selecting nodes to be dropped-out with a given probability (20%)
# We are using dropout to prevent the model from overfitting
# return_sequences = Boolean. Whether to return the last output. in the output sequence, or the full sequence.
# units = Positive integer, dimensionality of the output space.
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

# Prints summary of the model
model.summary()
# plot_model(model, to_file='../slike/model_plot.png', show_shapes=True, show_layer_names=True)

# Configures the model for training
model.compile(optimizer='adam', loss='mean_squared_error')

# Trains the model for a fixed number of epochs (iterations on a dataset) using the training data
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


""" TESTING MODEL ACCURACY ON EXISTING DATA """

# Loading Test Data
# Test data starts off where train data ended and ends on current day
test_start = dt.datetime(int(train_start_day[2]), int(train_start_day[1]), int(train_start_day[0]))
test_end = dt.datetime.now()

test_data = pdr.DataReader(company, 'yahoo', test_start, test_end)


# Making Predictions on Test Data
x_test = []

if use_mock_data:
    # Combined train and test data
    mock_data_test = []
    with open("step-data-test.txt", "r") as f:
        line = f.readline()

    for value in line.split(" "):
        if value:
            mock_data_test.append(float(value))

    df1 = pd.DataFrame(np.array(mock_data))
    df2 = pd.DataFrame(np.array(mock_data_test))
    total_dataset = pd.concat((df1, df2), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(mock_data_test) - timestep:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    for i in range(timestep, len(model_inputs)):
        x_test.append(model_inputs[i-timestep:i, 0])
    actual_prices = mock_data_test
else:   
    # Combined train and test data
    total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - timestep:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    for i in range(timestep, len(model_inputs)):
        x_test.append(model_inputs[i-timestep:i, 0])
    actual_prices = test_data['Adj Close'].values

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
# Inverse transform our scaled data
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting test predictions
plt.plot(actual_prices, color="black", label=f'Actual {company} price')
plt.plot(predicted_prices, color="red", label=f'Predicted {company} price')
plt.title(f'{company} Share Price')
plt.xlabel(f'Days passed from {test_start} up until {test_end}')
plt.ylabel(f'{company} Share Price (in american dollars $)')
plt.legend()
plt.show()


real_data = [model_inputs[len(model_inputs) - timestep:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'Prediction: {prediction}')