import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
import os
import pandas as pd
from datetime import datetime
import random
from scipy import ndimage

from common import *
from config import *


def random_flip(image, steering_angle):
    if bernoulli.rvs(0.5):
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle


def random_brightness(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def random_rotate(image):
    rotate = random.uniform(-1, 1)
    image = ndimage.rotate(image, rotate, reshape=False)

    return image


def augment_image(image, steering_angle):
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)
    image = random_rotate(image)

    return image, steering_angle


def generate_dataset(csv_path, img_path, output_csv_path, output_img_path):
    data = load_csv_data(csv_path)

    new_data = []
    for index in range(len(data)):
        file_name, img, angle, speed = load_image(data, img_path, index)
        new_image, new_angle = augment_image(img, angle)

        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        new_file_name = timestamp + "_" + file_name + ".jpg"

        cv2.imwrite(os.path.join(output_img_path, new_file_name), cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
        new_data.append([0, 0, 0, 0, 0, new_file_name, new_angle, 0, speed, 0, 0, 0])

    data = pd.DataFrame(new_data, columns=["index", "timestamp", "width", "height",
                                           "frame_id", "filename", "angle", "torque",
                                           "speed", "lat", "long", "alt"])
    data.to_csv(output_csv_path, mode='a', index=False, header=False)

# generate_dataset(DATASET_CSV, DATASET_DATA, "driving_log.csv", "IMG/")
