import tensorflow as tf
from keras import Model
from keras.layers import *

from raccoon_spotter.models.components import *


def conv_block(kernel):
    return tf.keras.Sequential(
        [
            Conv2D(32, kernel_size=(kernel, kernel), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(kernel, kernel), activation="relu"),
            SpatialPyramidPooling([1, 2, 4, 8]),
            BayesDense(1000, activation="relu"),
        ]
    )


def build_model(name: str = __name__.split(sep=".")[-1]):
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = Add()(
        [
            conv_block(1)(inputs),
            conv_block(3)(inputs),
            conv_block(5)(inputs),
        ]
    )
    x = BayesDense(128, activation="relu")(x)
    x = BayesDense(4, activation="relu")(x)
    return Model(inputs=inputs, outputs=x, name=name)
