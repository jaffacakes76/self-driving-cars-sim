import matplotlib.pyplot as plt
import matplotlib
import random
import os
import pickle

from common import *
from config import *

matplotlib.use("Agg")


def visualize_steering_distribution(data):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.hist(data['angle'], bins=75)
    ax.set_xlabel('Угао скретања', fontsize=12)
    ax.set_ylabel('Број слика', fontsize=12)

    plt.show()
    plt.savefig("steering_distribution.png")


def visualize_preprocessing(data, img_path):
    index = random.randrange(len(data))
    _, img, _, _ = load_image(data, img_path, index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Изворна слика', fontsize=12)
    axes[1].set_title('Препроцесирана слика', fontsize=12)

    axes[0].imshow(img)

    img = preprocess_img(img, clr="YUV")

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


# raw_data = load_csv_data(DATASET_CSV)
# visualize_steering_distribution(raw_data)
# visualize_preprocessing(raw_data, DATASET_DATA)
# visualize_loss("models/generated_dataset/model_1/YUV/hist_history.p", "models/generated_dataset/model_2/YUV/hist_history.p", "models/generated_dataset/model_3/YUV/hist_history.p")
