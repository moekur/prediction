from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, BatchNormalization, LeakyReLU, AveragePooling2D, concatenate
import cv2
import csv


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
        if (i > 10000):
            break
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

# Input layer
input_layer = Input(shape=(150,72, 3))  # According to image dimensions

# Initial convolutional layer
initial_conv = Conv2D(64, (3, 3), padding='same')(input_layer)
initial_bn = BatchNormalization()(initial_conv)
initial_relu = LeakyReLU()(initial_bn)

# Line 1
line1_maxpool1 = MaxPooling2D(pool_size=(2, 2))(initial_relu)
line1_conv1 = Conv2D(32, (3, 3), padding='same')(line1_maxpool1)
line1_bn1 = BatchNormalization()(line1_conv1)
line1_relu1 = LeakyReLU()(line1_bn1)
line1_maxpool2 = MaxPooling2D(pool_size=(2, 2))(line1_relu1)
line1_conv2 = Conv2D(32, (3, 3), padding='same')(line1_maxpool2)

# Line 2
line2_avgpool1 = AveragePooling2D(pool_size=(2, 2))(initial_relu)
line2_conv1 = Conv2D(32, (3, 3), padding='same')(line2_avgpool1)
line2_bn1 = BatchNormalization()(line2_conv1)
line2_relu1 = LeakyReLU()(line2_bn1)
line2_avgpool2 = AveragePooling2D(pool_size=(2, 2))(line2_relu1)
line2_conv2 = Conv2D(32, (3, 3), padding='same')(line2_avgpool2)


# Combine the inputs
combined = concatenate([line1_conv2, line2_conv2], axis=-1)


# Common path
common_bn = BatchNormalization()(combined)
common_maxpool = MaxPooling2D(pool_size=(2, 2))(common_bn)
common_conv = Conv2D(32, (3, 3), padding='same')(common_maxpool)
common_bn2 = BatchNormalization()(common_conv)
common_relu = LeakyReLU()(common_bn2)

# Flatten operation
flattened = Flatten()(common_relu)

# Fully connected layers
fc1 = Dense(units=100, activation='relu')(flattened)
fc1_relu = LeakyReLU()(fc1)
fc2 = Dense(units=10, activation='relu')(fc1_relu)
fc2_relu = LeakyReLU()(fc2)
fc3 = Dense(units=1, activation='relu')(fc2_relu)

# Add a Dropout layer to avoid over fitting 
dropout_rate = 0.2  
dropout_layer = Dropout(rate=dropout_rate)(fc3)

# Output layer for regression (linear activation function)
output_layer = Dense(units=1, activation='linear')(dropout_layer)

model = Sequential()

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with mean squared error loss (for regression)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_mae',  # or 'val_accuracy' based on preference
                               patience=4,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore the best weights when monitoring metric has stopped improving


# Display the model summary
model.summary()
print(f"X has: {len(np.array(X_train))} and shape: {np.array(X_train).shape}")
print(f"y has: {len(y_train)} and shape: {y_train.shape}")

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)), callbacks=[early_stopping])

model.save("rgb.keras")

prediction = model.predict(np.array(X_test))

plt.plot(prediction,label = "Prediction")
plt.plot(y_test,label = "Actual")
plt.legend()
plt.show()

