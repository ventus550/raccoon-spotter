import cv2
import numpy as np


def reshape_image_arrays(image_arrays: np.ndarray) -> np.ndarray:
    GRAYSCALE_CHANNELS = 2
    reshaped_arrays = []
    for image_array in image_arrays:
        if len(image_array.shape) == GRAYSCALE_CHANNELS:
            image_gray2rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            reshaped_arrays.append(image_gray2rgb)
        else:
            reshaped_arrays.append(image_array)
    reshaped_array = np.array(reshaped_arrays, dtype=object)
    return reshaped_array


def resize_image_arrays(reshaped_array: np.ndarray):
    pass
