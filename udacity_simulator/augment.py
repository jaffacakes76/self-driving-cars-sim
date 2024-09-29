import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
import os
import pandas as pd
from datetime import datetime
import random
from random import shuffle
from scipy import ndimage

from common import *
from config import *


def random_shear(image, steering_angle, shear_range=200):
    if bernoulli.rvs(0.9):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)

        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        steering_angle += dsteering

    return image, steering_angle


def random_flip(image, steering_angle):
    if bernoulli.rvs(0.5):
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle


def random_brightness(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def augment_image(image, steering_angle):
    image, steering_angle = random_shear(image, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)

    return image, steering_angle


def generate_dataset(csv_path, img_path, output_csv_path, output_img_path):
    data = load_csv_data(csv_path)
    new_data = []

    for index in range(len(data)):
        paths = []
        for position in ["center", "left", "right"]:
            img, angle, file_name = load_image(data, img_path, index, position)
            new_image, new_angle = augment_image(img, angle)

            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            new_file_name = timestamp + "_" + position + ".jpg"

            cv2.imwrite(os.path.join(output_img_path, new_file_name), cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
            paths.append(new_file_name)

        new_data.append([paths[0], paths[1], paths[2], new_angle, 0, 0, 0])

    data = pd.DataFrame(new_data, columns=["center", "left", "right", "steering", "throttle", "brake", "speed"])
    data.to_csv(output_csv_path, mode='a', index=False, header=False)


# generate_dataset(DATASET_CSV, DATASET_DATA, "driving_log.csv", "IMG/")
