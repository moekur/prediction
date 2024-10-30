import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


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
df_selected = df[selected_features + ['hour', 'day_of_week','date']]

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
#print(X)
#print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the NARX model
n_features = X_train.shape[2]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, n_features)))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Make predictions for the next hour
last_hour_data = X_test[-1:, :, :]
predicted_power = model.predict(last_hour_data)
#print(scaler.inverse_transform(predicted_power))
predicted_power = scaler.inverse_transform(np.concatenate([last_hour_data[:, -1, 1:], predicted_power], axis=1))

print(f'Predicted Power for the Next Hour: {predicted_power}')
# Testing the model
#prediction = model.predict(X_test)

#for p, y in zip(prediction,y_testing):
#    print(f"Prediction {p} => Actual {y}")

#plt.plot(scaler.inverse_transform(prediction),label = "Prediction")
#plt.plot(scaler.inverse_transform(y_test),label = "Actual")
#plt.legend()
#plt.show()