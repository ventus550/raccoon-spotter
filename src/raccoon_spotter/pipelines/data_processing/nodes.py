from typing import Dict

import cv2
import numpy as np


def reshape_image_arrays(image_arrays: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Add a color channel for greyscale images.

    Parameters:
    - image_arrays (np.ndarray): A numpy array containing image arrays, x for images, y for labels.

    Returns:
    - reshaped_dict (str, np.ndarray): A dictionary containing x (reshaped array of images) and y (labels).
    """
    GRAYSCALE_CHANNELS = 2
    RGB_CHANNELS = 3
    reshaped_arrays = []
    for img_array in image_arrays["x"]:
        if len(img_array.shape) == GRAYSCALE_CHANNELS:
            converted_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            reshaped_arrays.append(converted_array)
        elif len(img_array.shape) != RGB_CHANNELS:
            raise ValueError(
                f"Image array does not have three dimensions: {img_array.shape}"
            )
        else:
            reshaped_arrays.append(img_array)
    return {
        "x": np.array(reshaped_arrays, dtype=object),
        "y": image_arrays["y"],
    }


def resize_image_arrays(
    resize_image_config: dict, image_array: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Resize an image array to the specified dimensions.

    Args:
    - image_array (np.ndarray): The image array to be resized.
    - target_width (int): The target width for resizing (default: 600).
    - target_height (int): The target height for resizing (default: 400).

    Returns:
     - A dictionary containing x (resized image array) and y (labels).
    """
    resized_arrays = []
    target_width = resize_image_config["target_width"]
    target_height = resize_image_config["target_height"]
    for img in image_array["x"]:
        resized_img = cv2.resize(img, (target_width, target_height))
        # resized_img = resized_img.astype(float)
        resized_arrays.append(resized_img)
    return {"x": np.stack(resized_arrays, axis=0), "y": image_array["y"]}
