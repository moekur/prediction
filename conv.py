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
        sequences.append(scaler.fit_transform(data.iloc[:,:-1])) # return input data for one day to be represented in an image
        y = np.max(scaler.fit_transform(np.reshape(data.iloc[:,-1],(-1, 1))))
        target.append(y)
        #target.append(np.mean(data.iloc[:,-1]))

#print(sequences[0])

# Convert the time series data to RGB image
images = toImages(sequences,header,"./images/")

# Write the images on disk    
concatenated_images = np.concatenate(images, axis=1)
cv2.imwrite("concatenated.jpg", concatenated_images)

# Split the data for training and testing
x_training, x_testing, y_training, y_testing = train_test_split(
    images, target, test_size=0.2
)

# Input layer
input_layer = Input(shape=(150,72, 3))  # According to image dimensions

model = Sequential()

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

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with mean squared error loss (for regression)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_mae',  
                               patience=4,  
                               restore_best_weights=True)  


# Display the model summary
model.summary()
print(f"X has: {len(np.array(x_training))} and shape: {np.array(x_training).shape}")
print(f"y has: {len(np.array(y_training))} and shape: {np.array(y_training).shape}")
print(np.array(x_training)[0])
# Train the model
#model.fit(np.array(x_training), np.array(y_training), validation_data=(np.array(x_testing), np.array(y_testing)),epochs=40, batch_size=32, callbacks=[early_stopping])

# Testing the model
#prediction = model.predict(np.array(x_testing))
#for p, y in zip(prediction,y_testing):
#    print(f"Prediction {p} => Actual {y}")

#plt.plot(prediction,label = "Prediction")
#plt.plot(y_testing,label = "Actual")
#plt.legend()
#lt.show()

# Save the model with all configs
#model.save("solar.keras")