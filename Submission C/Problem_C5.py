# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model 
# should be trained to predict the next 24 observations of the 7 variables.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
#
# Desired MAE < 0.1 on the normalized dataset.
# ============================================================================================

import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mae = logs.get('mae')
        if mae is not None and mae < 0.1:
            print("\nReached Desired MAE so Finish.........")
            self.model.stop_training = True

# This function downloads and extracts the dataset to the directory that contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change the URL)
def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_past + n_future))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda window: (window[:-n_past], window[-n_past:, :1]))
    return dataset.batch(batch_size).prefetch(1)

# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def solution_C5():
    # Downloads and extracts the dataset to the directory that contains this file.
    download_and_extract_data()
    # Reads the dataset from the csv.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = len(df.columns)# YOUR CODE HERE

    # Normalizes the data
    # DO NOT CHANGE THIS
    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    x_train = data[:split_time]
    x_valid = data[split_time:]

    # DO NOT CHANGE THIS
    BATCH_SIZE = 32  
    N_PAST = 24 # Number of past time steps based on which future observations should be predicted
    N_FUTURE = 24  # Number of future time steps which are to be predicted.
    SHIFT = 1  # By how many positions the window slides to create a new window of observations.

    # Code to create windowed train and validation datasets.
    # Complete the code in windowed_dataset.
    train_set = windowed_dataset(
        series=x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT
    )
    valid_set = windowed_dataset(
        series=x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT
    )

    # Code to define your model.
    model = tf.keras.models.Sequential([
        # Whatever your first layer is, the input shape will be (N_PAST = 24, N_FEATURES = 7)
        tf.keras.layers.InputLayer(input_shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(2 * N_FEATURES, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(2 * N_FEATURES, return_sequences=True)
        ),
        # YOUR CODE HERE
        tf.keras.layers.Dense(N_FUTURE),
    ])

    # Code to train and compile the model
    customCallback = myCallback()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=["mae"])

    model.fit(train_set, epochs=1000,
              callbacks=[customCallback],
              validation_data=valid_set)
    # YOUR CODE HERE
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C5()
    model.save("model_C5.h5")