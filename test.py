from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, BatchNormalization, LeakyReLU, AveragePooling2D, concatenate
import cv2
import csv

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from myutilities import getAllFiles, toImages
from myutilities import isIn
from keras.callbacks import EarlyStopping


# Assuming df is your DataFrame with timestamp, active energy, voltage, current, wind speed, diffuse radiation, temperature, and power columns
header = ['timestamp','Weather_Temperature_Celsius','Radiation_Diffuse_Tilted','Wind_Speed','Active_Power']
df = pd.read_csv("DKAAC.csv").dropna()
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Feature selection
selected_features = ['Active_Power', 'Current_Phase_Average', 'Wind_Speed', 'Radiation_Diffuse_Tilted', 'Weather_Temperature_Celsius']

# Add additional features like hour of the day, day of the week, etc.
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['date'] = df['timestamp'].dt.dayofyear
df = df.drop(columns=['timestamp'])

# Select relevant features
selected_header = selected_features + ['hour', 'day_of_week','date']
df_selected = df[selected_header]

# Normalize the data
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_selected)

# Function to create sequences for NARX model
def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length - 1):  # Adjusted for the delay
        seq = data[i:i + seq_length, :]
        label = np.mean(data[i + seq_length + 1, 0])  # Power is the target variable, shifted by 1 step
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

# Choose sequence length
seq_length = 12  # Assuming hourly data

# Create sequences and target variable
X, y = create_sequences(df_normalized, seq_length)

X_images = toImages(X, selected_header, "./delayed_images/")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.05, shuffle=False)

# Write the images on disk
sub_images = []
for i in range(1000):
    sub_images.append(X_images[i])
concatenated_images = np.concatenate(sub_images, axis=1)
cv2.imwrite("./delayed_images/concatenated.jpg", concatenated_images)


# Reload the model
print("Reload the model....")
model = tf.keras.saving.load_model("rgb.keras")

prediction = model.predict(X_test)
for p, y in zip(prediction,y_test):
    print(f"Prediction {p} => Actual {y}")

plt.plot(prediction,label = "Prediction")
plt.plot(y_test,label = "Actual")
plt.legend()
plt.show()