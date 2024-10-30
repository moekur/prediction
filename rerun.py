import pandas as pd
from myutilities import toImages
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


test_header = ['timestamp','Weather_Temperature_Celsius','Radiation_Diffuse_Tilted','Wind_Speed','Active_Power']
test_df = pd.read_csv("DKAAC.csv").loc[:,test_header].dropna()
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

scaler = StandardScaler()
#test_df.set_index('timestamp')
test_sequences = []
test_target = []
for day, group in test_df.groupby(test_df.timestamp.dt.date):
    group['Active_Power'] =  group['Active_Power'].astype(float)
    y = np.max(scaler.fit_transform(np.reshape(group.iloc[:,-1],(-1, 1))))
    test_target.append(y)
    group.drop('timestamp', axis=1, inplace=True)
    group.drop('Active_Power', axis=1, inplace=True)
    test_sequences.append(group)


test_images = toImages(test_sequences,['Weather_Temperature_Celsius','Radiation_Diffuse_Tilted','Wind_Speed','Active_Power'])


#test_target = scaler.fit_transform(np.reshape(test_target,(-1, 1)))

# Write test images on disk    
#test_concatenated_images = np.concatenate(test_images, axis=1)
#cv2.imwrite("test_concatenated.jpg", test_concatenated_images)

print(len(test_target))
print(len(test_sequences))

# Split the data for training and testing
x_training, x_testing, y_training, y_testing = train_test_split(
    test_images, test_target, test_size=0.4
)
# Reload the model
print("Reload the model....")
reloaded_model = tf.keras.saving.load_model("solar.keras")

# Display the model summary
reloaded_model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_mae',  # or 'val_accuracy' based on preference
                               patience=4,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore the best weights when monitoring metric has stopped improving


# Train the model
#reloaded_model.fit(np.array(x_training), np.array(y_training), validation_data=(np.array(x_testing), np.array(y_testing)),epochs=40, batch_size=32, callbacks=[early_stopping])
print(x_training[0])
# Testing the model
prediction = reloaded_model.predict(np.array(x_testing))
for p, y in zip(prediction,y_testing):
    print(f"Prediction {p} => Actual {y}")

plt.plot(prediction,label = "Prediction")
plt.plot(y_testing,label = "Actual")
plt.legend()
plt.show()

