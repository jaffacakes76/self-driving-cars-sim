import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import cv2
import matplotlib
import pickle
import math
import sys

from common import *
from augment import *
from config import *
from more_itertools import unique_everseen

matplotlib.use("Agg")


def visualize_steering_distribution(data):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.hist(data['steering'], bins=75)
    ax.set_xlabel('Угао скретања', fontsize=12)
    ax.set_ylabel('Број слика', fontsize=12)

    plt.show()
    plt.savefig("steering_distribution.png")


def visualize_preprocessing(img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Изворна слика', fontsize=12)
    axes[1].set_title('Препроцесирана слика', fontsize=12)

    axes[0].imshow(img)

    img = preprocess_img(img)

    axes[1].imshow(img)

    plt.show()
    plt.savefig("preprocessing.png")


def visualize_loss(file_name_1, file_name_2, file_name_3):
    file1 = open(file_name_1, 'rb')
    history1 = pickle.load(file1)

    file2 = open(file_name_2, 'rb')
    history2 = pickle.load(file2)

    file3 = open(file_name_3, 'rb')
    history3 = pickle.load(file3)

    plt.plot(history1['loss'])
    plt.plot(history1['val_loss'])

    plt.plot(history2['loss'])
    plt.plot(history2['val_loss'])

    plt.plot(history3['loss'])
    plt.plot(history3['val_loss'])

    plt.title('Губитак модела', fontsize=12)
    plt.ylabel('Губитак', fontsize=12)
    plt.xlabel('Епоха', fontsize=12)

    plt.legend(['Модел 1 тренирање', 'Модел 1 валидација', 'Модел 2 тренирање', 'Модел 2 валидација', 'Модел 3 тренирање', 'Модел 3 валидација'], loc='upper right')

    plt.show()
    plt.savefig("loss.png")


def visualize_random_samples(data_path, data):
    number_of_images = len(data['center'])
    camera_positions = ['left', 'center', 'right']
    titles = ['лева', 'централна', 'десна']

    fig, axes = plt.subplots(3, 3, figsize=(12, 6))
    for pos, ax in zip(titles, axes[0, :]):
        ax.set_title(pos + ' камера')
    for ax in axes:
        img_index = random.randrange(number_of_images)
        for a, pos in zip(ax, camera_positions):
            img, _, _ = load_image(data, data_path, img_index, pos)
            a.imshow(img)
            a.axis('off')
    plt.show()
    plt.savefig("random_samples.png")


def visualize_random_shear(img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original')
    axes[1].set_title('Sheared')

    axes[0].imshow(img)
    axes[0].axis('off')

    img, angle = random_shear(img, 0.0)
    print(angle)

    axes[1].imshow(img)
    axes[1].axis('off')

    plt.show()
    plt.savefig("random_shear.png")


def visualize_random_flip(img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original')
    axes[1].set_title('Flipped')

    axes[0].imshow(img)
    axes[0].axis('off')

    img, _ = random_flip(img, 0.0)

    axes[1].imshow(img)
    axes[1].axis('off')

    plt.show()
    plt.savefig("random_flip.png")


def visualize_random_brightness(img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original')
    axes[1].set_title('Brightened')

    axes[0].imshow(img)
    axes[0].axis('off')

    img = random_brightness(img)

    axes[1].imshow(img)
    axes[1].axis('off')

    plt.show()
    plt.savefig("random_brightness.png")


# raw_data = load_csv_data(DATASET_CSV)
# index = random.randrange(len(raw_data))
# position = np.random.choice(['left', 'center', 'right'])
# image, _, _ = load_image(raw_data, DATASET_DATA, index, position)

# visualize_steering_distribution(raw_data)
# visualize_preprocessing(image)
# visualize_random_samples(DATASET_DATA, raw_data)
# visualize_loss("models/udacity_dataset/novi/model_1/YUV/hist_history.p", "models/udacity_dataset/novi/model_2/YUV/hist_history.p", "models/udacity_dataset/novi/model_3/YUV/hist_history.p")

# visualize_random_shear(image)
# visualize_random_flip(image)
# visualize_random_brightness(image)


def remove_duplicates_csv(old_csv_path, new_csv_path):
    with open(old_csv_path, 'r') as f, open(new_csv_path, 'w') as out_file:
        out_file.writelines(unique_everseen(f))


def visualize_trajectory(csv_path1, csv_path2):
    data1 = pd.read_csv(csv_path1, header=None, names=["X", "Y", "Z"])

    data2 = pd.read_csv(csv_path2, header=None, names=["X", "Y", "Z"])

    plt.figure(figsize=(15, 5))

    plt.plot(data1['X'], data1['Z'], linewidth=2.0, label="Предиктована путања")
    # plt.plot(data2['X'], data2['Z'], linewidth=2.0, label="Тачна путања")
    plt.legend(loc="upper left")

    plt.show()
    plt.savefig("diagram.png")


def calculate_distance(csv_path1, csv_path2):
    data1 = pd.read_csv(csv_path1, header=None, names=["X", "Y", "Z"])
    data2 = pd.read_csv(csv_path2, header=None, names=["X", "Y", "Z"])

    data_frame = []
    for i in range(5000, 9258):
        point1 = (data1.iloc[i]["X"], data1.iloc[i]["Z"])
        dist1 = math.dist((0, 0), point1)
        points = [((data2.iloc[j]["X"], data2.iloc[j]["Z"]), math.dist((0, 0), (data2.iloc[j]["X"], data2.iloc[j]["Z"]))) for j in range(len(data2))]
        min_dist = sys.maxsize
        final_min = None
        for j in range(len(points)):
            dist = math.dist(point1, points[j][0])
            if dist < min_dist:
                min_dist = dist
                final_min = dist
                if points[j][1] > dist1:
                    final_min = - final_min

        data_frame.append([final_min])
        print(i)

    data = pd.DataFrame(data_frame, columns=["dist"])
    data.to_csv("distance.csv", mode='a', index=False, header=False)


def calculate_MSE(csv_path):
    data = pd.read_csv(csv_path, header=None, names=["dist"])
    y = np.array(data["dist"])
    plt.figure(figsize=(15, 5))
    plt.plot(y, linewidth=1.0)
    plt.show()
    plt.savefig("distance.png")
    MSE = sum(y*y) / len(y)
    print(MSE)


# remove_duplicates_csv("./output/generated_dataset/model_3/RGB/koord.csv", "novi.csv")
# visualize_trajectory("./output/generated_dataset/model_3/RGB/koord.csv", "./output/koord.csv")
# calculate_distance("./output/generated_dataset/model_3/RGB/koord.csv", "./output/koord.csv")
# calculate_MSE("distance.csv")
