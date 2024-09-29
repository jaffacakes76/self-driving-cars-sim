import pickle
import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D

matplotlib.use("Agg")


def create_base_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Conv2D(64, (3, 3)))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    return model


def create_model_1():
    model = Sequential()

    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2), input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model


def create_model_2():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model


def create_model_3():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model


def save_history(history):
    hist_history = history.history
    pickle.dump(hist_history, open("hist_history.p", "wb"))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("loss.png")

