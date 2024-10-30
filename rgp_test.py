from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from myutilities import toImages
import matplotlib.pyplot as plt


# ... (Your import statements)

# Load and preprocess data
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
selected_header = selected_features + ['hour', 'day_of_week', 'date']
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
        if (i > 1000):
            break
        
    return np.array(sequences), np.array(target)

X, y = [],[]
# Create sequences and target variable
X, y = create_sequences(df_normalized, 12)

# Convert sequences to images
X_images = toImages(X, selected_header, "./delayed_images/")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 72, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(optimizer="adam", loss='mse', metrics=['mae'])  # Use mean squared error for regression

print(f"X has: {len(np.array(X_train))} and shape: {np.array(X_train).shape}")
print(f"y has: {len(np.array(y_train))} and shape: {np.array(y_train).shape}")
print(np.array(X_train)[0])
# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=30, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))
prediction = model.predict(np.array(X_test))

plt.plot(np.array(prediction),label = "Prediction")
plt.plot(np.array(y_test),label = "Actual")
plt.legend()
plt.show()
