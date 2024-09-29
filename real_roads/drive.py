import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
import shutil
import pandas as pd
from statistics import mean
import matplotlib
import tensorflow as tf

from common import *
from config import *


matplotlib.use("Agg")


def test(model, csv_path, images_path):
    raw_data = pd.read_csv(csv_path, names=["frame_id", "angle", "public"], dtype={'frame_id': str, 'angle': float,
                                                                                   'public': int})
    data_frame = []
    for i in range(len(raw_data)):
        image = plt.imread(images_path + raw_data.iloc[i]["frame_id"] + ".jpg")
        image = preprocess_img(image, clr="YUV")

        ref_angle = raw_data.iloc[i]["angle"]
        pred_angle = float(model.predict(image[None, :, :, :], batch_size=1))

        data_frame.append([ref_angle, pred_angle])

    data = pd.DataFrame(data_frame, columns=["ref_angle", "pred_angle"])
    data.to_csv("result.csv", mode='a', index=False, header=False)


def show_difference(csv_path):
    raw_data = pd.read_csv(csv_path, header=None, names=["ref_angle", "pred_angle"])

    y1 = np.array(raw_data["ref_angle"])
    y2 = np.array(raw_data["pred_angle"])

    last_steering_angle = None
    for i in range(len(y2)):
        if last_steering_angle is None:
            last_steering_angle = y2[i]
        steering_angle = 0.1 * y2[i] + (1 - 0.1) * last_steering_angle
        last_steering_angle = steering_angle
        y2[i] = steering_angle

    MSE = 1 / len(raw_data) * sum((y1 - y2) * (y1 - y2))
    print(MSE)

    plt.figure(figsize=(15, 5))
    plt.ylabel('Угао скретања', fontsize=12)
    plt.xlabel('Редни број фрејма', fontsize=12)
    plt.plot(y2, linewidth=2.0, label="Предиктоване вредности")
    plt.plot(y1, linewidth=2.0, label="Стварне вредности")
    plt.legend(loc="upper right")

    plt.show()
    plt.savefig("diagram.png")


def combine_images(image, wheel):
    wheel = cv2.resize(wheel, (120, 120), cv2.INTER_AREA)
    image_pil = Image.fromarray(image)
    wheel_pil = Image.fromarray(wheel)
    image_pil.paste(wheel_pil, (0, 0))

    return np.asarray(image_pil)


def gen_image(full_image, angle, steer_img, image_folder):
    rows, cols, _ = steer_img.shape
    degrees = angle * 180 / 3.14159265

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    wheel = cv2.warpAffine(steer_img, M, (cols, rows))
    result = combine_images(full_image, wheel)

    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_filename = os.path.join(image_folder, timestamp)
    cv2.imwrite('{}.jpg'.format(image_filename), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def save(test_csv_path, images_path, angles_csv_path):
    steer_img = cv2.imread(STEERING_WHEEL_IMG)

    image_folder = "./output/images"
    if image_folder != '':
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        else:
            shutil.rmtree(image_folder)
            os.makedirs(image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    raw_data = pd.read_csv(test_csv_path, header=None, names=["frame_id", "angle", "public"],
                           dtype={'frame_id': str, 'angle': float,
                                  'public': int})

    angles = pd.read_csv(angles_csv_path, header=None, names=["ref_angle", "pred_angle"])
    i = 0

    last_steering_angle = None
    while i < len(raw_data):
        image = plt.imread(images_path + raw_data.iloc[i]["frame_id"] + ".jpg")
        angle = angles.iloc[i]["pred_angle"]
        if last_steering_angle is None:
            last_steering_angle = angle
        angle = 0.1 * angle + (1 - 0.1) * last_steering_angle
        last_steering_angle = angle
        gen_image(image, angle, steer_img, image_folder)

        i += 1


# model = load_model('./models/generated_dataset/novi/model_3/YUV/model.h5')
# test(model, TEST_CSV, TEST_DATA)
# show_difference("result.csv")
# save(TEST_CSV, TEST_DATA, "output/generated_dataset/dropout-all/RGB/check.csv")
