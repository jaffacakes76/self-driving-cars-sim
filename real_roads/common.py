import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from config import *


def load_csv_data(csv_path):
    data = pd.read_csv(csv_path, header=None, names=["index", "timestamp", "width", "height",
                                                     "frame_id", "filename", "angle", "torque",
                                                     "speed", "lat", "long", "alt"])
    return data


def split_dataset(csv_path, train_csv_path, valid_csv_path):
    data = load_csv_data(csv_path)
    train_data, valid_data = train_test_split(data, test_size=0.2)

    train_data.to_csv(train_csv_path, mode='a', index=False, header=False)
    valid_data.to_csv(valid_csv_path, mode='a', index=False, header=False)


def load_image(data, images_path, index):
    file_name = data.iloc[index]["filename"]
    image = plt.imread(os.path.join(images_path, file_name))
    angle = data.iloc[index]['angle']
    speed = data.iloc[index]['speed']

    return file_name, image, angle, speed


def camera_adjust(angle, speed, file_name):
    if "center" in file_name:
        return angle

    reaction_time = 2.0  # seconds
    x = 20.0  # inches
    y = speed * reaction_time * 12.0  # (ft/s)*s*(12 in/ft) = inches

    angle_adj = np.arctan(float(x) / y)  # radians

    if "left" in file_name:
        angle_adj = -angle_adj

    angle = angle_adj + angle

    return angle


def preprocess_img(image, clr="RGB"):
    image = image[100:image.shape[0], 0:image.shape[1]]
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)

    if clr == "YUV":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    return image
