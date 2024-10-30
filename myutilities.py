import re
from os import walk
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.colors import hsv_to_rgb


# Checks if a certain key is on the the provided list (lst)

def isIn(key, lst):
    for item in lst:
        if re.match(key +"*",item) is not None:
            return item
    return None


# Search deeply on path for all files with this specific extensions (e.g. *.csv)

def getAllFiles(path,extension):
    res = []
    for (dir_path, dir_names, file_names) in walk(path):
        for file in file_names:
            if file.endswith(extension):
                res.append(dir_path + "/" + file)
    if res:
        return res
    else:
        return None
# Get sequences of array of time series data and return it as array of RGB images
images = []
# Define the resolution of the output image
def toImages(sequences, header, path, output_resolution = (72, 150)):
    i = 0
    for sequence in sequences:
#        print(sequence.shape)
        hsv_data = np.zeros((sequence.shape[0],sequence.shape[1], 3), dtype=np.float32)
 #       print(hsv_data)
        #print(str(np.shape(hsv_data)) + " => " + str(np.shape(normalized_data[header[0]].values.reshape(-1, 1))))
        hsv_data[:, :, 0] = sequence  # Hue
        hsv_data[:, :, 1] = 1.0  # Saturation
        hsv_data[:, :, 2] = 1.0
        
        # Transpose the data to adjust the dimensions
        #hsv_data = hsv_data.transpose(1, 0, 2)

        # Convert HSV to RGB
        #rgb_data = cv2.cvtColor(hsv_data, cv2.COLOR_HSV2RGB)
        rgb_data = hsv_to_rgb(hsv_data)

        # Convert to 8-bit depth
        rgb_data = (rgb_data * 255).astype(np.uint8)
        # Convert HSV to RGB
        
        # Reshape to the desired resolution
        rgb_image = cv2.resize(rgb_data, (output_resolution[0], output_resolution[1]), interpolation=cv2.INTER_AREA)
        if ( i < 0 ):
            print(hsv_data)
            print("-----------------------")
            cv2.imwrite(path + "sequence_" + str(i) + ".jpg", rgb_image)
        images.append(rgb_image)
        i = i + 1
    return images
