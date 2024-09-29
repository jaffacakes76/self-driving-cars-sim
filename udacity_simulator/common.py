import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split

from config import *


def load_csv_data(csv_path):
    data = pd.read_csv(csv_path, header=None,
                       names=["center", "left", "right", "steering", "throttle", "brake", "speed"])

    return data


def split_dataset(csv_path, train_csv_path, valid_csv_path):
    data = load_csv_data(csv_path)
    train_data, valid_data = train_test_split(data, test_size=0.2)

    train_data.to_csv(train_csv_path, mode='a', index=False, header=False)
    valid_data.to_csv(valid_csv_path, mode='a', index=False, header=False)


def load_image(data, img_path, index, position):
    file_name = data.iloc[index][position].split('/')[-1]
    img = plt.imread(os.path.join(img_path, file_name))
    angle = data.iloc[index]['steering']

    return img, angle, file_name


def camera_adjust(angle, position):
    if "center" in position:
        return angle
    if "left" in position:
        return angle + STEERING_COEFFICIENT
    if "right" in position:
        return angle - STEERING_COEFFICIENT


def preprocess_img(image, clr="RGB"):
    image = image[55:140, 0:image.shape[1]]
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)

    if clr == "YUV":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    return image
