import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, BatchNormalization, LeakyReLU, AveragePooling2D, concatenate
import cv2
import csv
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from myutilities import getAllFiles, toImages
from myutilities import isIn
from keras.callbacks import EarlyStopping

subset = ["SEWSAmbientTemp_C", "RefCell1_Wm2", "WindSpeedAve_ms", "InvPDC_kW"]
path="/Users/eissa/AICC/onemin-Ground-2017/datasets1"

files = getAllFiles(path,".csv")

# Set dataset header based on the first file header
header = []
header_df = pd.read_csv(files[0])

for column in subset:
    column_name = isIn(column,header_df)
    if column_name is not None:
        header.append(column_name)


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

# Load data & Create sequences
scaler = StandardScaler()
target = []
sequences = []
for file in files:
    #sequences = pd.concat([sequences,pd.read_csv(file)[header]],ignore_index=True)
    data = pd.read_csv(file).loc[:,header].dropna()
    #data = data[data.iloc[:, -1] >= 0] #InvPDC_kW_Avg
    #data = data[data.InvPDC_kW_Avg > 0]
    #print(len(data))
    #print(file + " #rows: " , len(data))
    # Fit the scaler on target and transform it
    if (len(data) > 0):
        #print(np.max(data.iloc[:,-1]))
        #print(file, " => ",np.mean(scaler.fit_transform(np.reshape(data.iloc[:,-1],(0, 24)))))
        sequences.append(data.iloc[:,:-1]) # return input data for one day to be represented in an image
        y = np.max(scaler.fit_transform(np.reshape(data.iloc[:,-1],(-1, 1))))
        target.append(y)
        #target.append(np.mean(data.iloc[:,-1]))

print(sequences.head())

# Create sequences and target variable
X, y = create_sequences(sequences, 12)

# Convert sequences to images
X_images = toImages(X, header, "./delayed_images/")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 72, 3)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(optimizer="adam", loss='mse', metrics=['mae'])  # Use mean squared error for regression

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

