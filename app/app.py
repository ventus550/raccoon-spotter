import base64
import os
import signal
import sys

import click
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from gevent.pywsgi import WSGIServer
from imports.components import *

# imports are the source of an external code that is copied during contenerization
from imports.utils.data_conversion import base64_to_pil
from imports.utils.data_visualization import roi
from tensorflow import keras


def catch(sig, frame):
    click.secho("Application terminated successfully.", fg="yellow")
    sys.exit(0)


signal.signal(signal.SIGINT, catch)

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.keras")

model = keras.models.load_model(
    MODEL_PATH, custom_objects={"SpatialPyramidPooling": SpatialPyramidPooling}
)


def model_predict(img, model):
    img = np.array(img.convert("RGB"))
    img = cv2.resize(img, (600, 400), interpolation=cv2.INTER_LINEAR)

    preds = np.squeeze(
        [model.predict(np.expand_dims(img, axis=0)) for _ in range(100)]
    ).astype(int)
    return np.mean(
        [roi(img, p, dim=0.2, rec=0) for p in preds], axis=0, dtype=int
    ).astype(np.uint8)

    preds = model.predict(np.expand_dims(img, axis=0))

    img_with_box = roi(img, preds[0].astype(int))
    return img_with_box


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    img = base64_to_pil(request.json["image"])

    img_with_box = model_predict(img, model)

    correct_color = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)

    _, img_encoded = cv2.imencode(".png", correct_color)
    img_bytes = img_encoded.tobytes()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify(image_base64=img_base64)


if __name__ == "__main__":
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    click.secho("Application running at http://localhost:5000.", fg="yellow")
    http_server.serve_forever()
