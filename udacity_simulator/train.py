import os
import numpy as np
from random import shuffle

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from common import *
from config import *
from train_functions import *


def generate_next_batch(data, img_path, batch_size):
    while True:
        x_batch, y_batch = [], []
        rnd_indices = list(range(0, len(data)))
        shuffle(rnd_indices)

        for index in rnd_indices:
            for position in ["center"]:
                img, angle, _ = load_image(data, img_path, index, position)

                new_image = preprocess_img(img)
                angle = camera_adjust(angle, position)

                x_batch.append(new_image)
                y_batch.append(angle)

                if len(x_batch) == batch_size:
                    yield np.array(x_batch), np.array(y_batch)
                    x_batch, y_batch = [], []


def train_model(train_csv_path, valid_csv_path, images_path):
    model = create_model_1()
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")

    train_data = load_csv_data(train_csv_path)
    valid_data = load_csv_data(valid_csv_path)

    checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='auto')

    stop = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    steps_per_epoch = int(len(train_data) / TRAIN_BATCH_SIZE)
    validation_steps = int(len(valid_data) / VALID_BATCH_SIZE)

    train_gen = generate_next_batch(train_data, images_path, TRAIN_BATCH_SIZE)
    validation_gen = generate_next_batch(valid_data, images_path, VALID_BATCH_SIZE)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch,
                                  epochs=EPOCHS,
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps,
                                  callbacks=[checkpoint, stop],
                                  verbose=1)

    model.save("model.h5", save_format="h5")
    save_history(history)


# split_dataset(DATASET_CSV, TRAIN_DATASET_CSV, VALID_DATASET_CSV)
train_model(TRAIN_DATASET_CSV, VALID_DATASET_CSV, DATASET_DATA)
