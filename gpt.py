import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from myutilities import getAllFiles
from myutilities import isIn
import matplotlib.pyplot as plt


# Assuming you have a dataset in a CSV file
# Replace 'your_dataset.csv' with your actual file name
#data = pd.read_csv('your_dataset.csv')
subset = ["SEWSAmbientTemp_C", "RefCell1_Wm2", "WindSpeedAve_ms", "InvPDC_kW"]
path="/Users/eissa/AICC/onemin-Ground-2017/datasets1"

files = getAllFiles(path,".csv")

# Set dataset header based on the first file header
header = []
header_df = pd.read_csv(files[0])
#print(header_df.head())
for column in subset:
    column_name = isIn(column,header_df)
    if column_name is not None:
        header.append(column_name)

print(header)
# Load data 
data = pd.DataFrame([])
for file in files:
    data = pd.read_csv(file)[header].dropna()
data.dropna(inplace=True)

#print(data.head())

# Assuming your dataset has columns 'solar_radiation', 'temperature', 'humidity', and 'output_power'
# Adjust column names accordingly based on your dataset
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the output power labels (assuming it's categorical)
label_encoder = LabelEncoder()
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output layer with linear activation for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'\nTest Loss (MSE): {loss:.4f}')
print(f'Test Mean Absolute Error (MAE): {mae:.4f}')

# Make predictions
predictions = model.predict(X_test)
#original_predictions = label_encoder.inverse_transform(predictions)
for p,a in zip(predictions,y_test):
    print(f"Prediction: {p} => Actual: {a}")

plt.plot(predictions,label="Prediction")
plt.plot(y_test,label="Actual")

plt.legend()
plt.show()

#=============
test_header = ['Weather_Temperature_Celsius','Radiation_Diffuse_Tilted','Wind_Speed','Active_Power']
test_df = pd.read_csv("DKAAC.csv").loc[:,test_header].dropna()

test_X = test_df.iloc[:,:-1]
test_y = test_df.iloc[:,-1]

# Standardize the features
test_X = scaler.fit_transform(test_X)

# Encode the output power labels (assuming it's categorical)
#label_encoder = LabelEncoder()
test_y = scaler.fit_transform(np.array(test_y).reshape(-1, 1))

# Split the dataset into training and testing sets
test_X_train, test_X_test, test_y_train, test_y_test = train_test_split(test_X, test_y, test_size=0.05, random_state=42)

model.fit(test_X_train, test_y_train, epochs=10, batch_size=32, validation_data=(test_X_test, test_y_test))

# Evaluate the model
loss, mae = model.evaluate(test_X_test, test_y_test)
print("====================New Dataset==================")
print(f'\nTest Loss (MSE): {loss:.4f}')
print(f'Test Mean Absolute Error (MAE): {mae:.4f}')

# Make predictions
predictions = model.predict(test_X_test)
#original_predictions = label_encoder.inverse_transform(predictions)
#for tp,ta in zip(predictions,test_y_test):
#    print(f"Prediction: {tp} => Actual: {ta}")

plt.plot(predictions,label="Prediction")
plt.plot(test_y_test,label="Actual")

plt.legend()
plt.show()
