import tensorflow as tf
from keras.layers import *

from raccoon_spotter.models.components.spatial_pyramid_pooling import (
    SpatialPyramidPooling,
)


def build_model(name: str = __name__.split(sep=".")[-1]):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None, None, 3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            SpatialPyramidPooling([1, 2, 4]),
            Dropout(0.5),
            Dense(4, activation="relu"),
        ],
        name=name,
    )
