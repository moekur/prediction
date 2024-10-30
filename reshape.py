import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Load your data
df = pd.read_csv("DKAAC.csv").dropna()

# Normalize the data
scaler = MinMaxScaler()

# Select relevant features (excluding the target variable Active_Power)
features = ['Weather_Temperature_Celsius', 'Radiation_Diffuse_Tilted', 'Wind_Speed']
target_variable = 'Active_Power'
data = df[features + [target_variable]]
scaled_data = scaler.fit_transform(data)
X = scaled_data[:, :-1]  # Features excluding the target
y = scaled_data[:, -1]   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression model with dropout
model = Sequential()
model.add(Dense(50, input_dim=len(features), activation='relu'))
model.add(Dropout(0.2))  # Add dropout with a dropout rate of 0.2
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))  # Add dropout with a dropout rate of 0.2
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_mae',  # or 'val_accuracy' based on preference
                               patience=4,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore the best weights when monitoring metric has stopped improving

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stopping])

# Plotting predictions and actual values
prediction = model.predict(X_test)

# Plotting loss during training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


plt.plot(prediction,label = "Prediction")
plt.plot(y_test,label = "Actual")
plt.legend()
plt.show()
