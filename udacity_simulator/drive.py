import base64
import numpy as np
from PIL import Image
from io import BytesIO
import os
import shutil
from datetime import datetime
from keras.models import load_model
import pandas as pd

import socketio  # real-time server
import eventlet.wsgi  # web server gateway interface
from flask import Flask  # web framework

from common import *
from config import *

sio = socketio.Server()
app = Flask(__name__)


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.05, 0.002)  # 0.1
set_speed = 5
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = data["speed"]
        img_string = data["image"]

        image = Image.open(BytesIO(base64.b64decode(img_string)))
        image_array = np.asarray(image)
        image_array = preprocess_img(image_array)

        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        throttle = controller.update(float(speed))

        send_control(steering_angle, throttle)

        if IMAGE_FOLDER != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(IMAGE_FOLDER, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer", data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        }, skip_sid=True)


if __name__ == '__main__':
    model = load_model(MODEL_PATH)

    if IMAGE_FOLDER != '':
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
        else:
            shutil.rmtree(IMAGE_FOLDER)
            os.makedirs(IMAGE_FOLDER)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
