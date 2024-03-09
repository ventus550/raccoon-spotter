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


def _letterbox_image(img, inp_dim):
    """resize image with unchanged aspect ratio using padding"""
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)
    canvas[
        (h - new_h) // 2 : (h - new_h) // 2 + new_h,
        (w - new_w) // 2 : (w - new_w) // 2 + new_w,
        :,
    ] = resized_image
    return canvas


def resize_image_arrays(
    resize_image_config: dict, image_array: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Resize the image while adjusting the corresponding bounding box.

    Args:
    - image_array (np.ndarray): The image array to be resized.
    - target_width (int): The target width for resizing (default: 600).
    - target_height (int): The target height for resizing (default: 400).

    Returns:
     - A dictionary containing x (resized image array) and y (adjusted labels).
    """
    resized_arrays = []
    original_image_sizes = []

    target_width = resize_image_config["target_width"]
    target_height = resize_image_config["target_height"]

    for img in image_array["x"]:
        original_size = img.shape[:2]
        original_image_sizes.append(original_size)
        resized_img = _letterbox_image(img, (target_width, target_height))
        resized_arrays.append(resized_img)

    resized_boxes = []
    for bbox, original_size in zip(image_array["y"], original_image_sizes):
        scale_x = target_width / original_size[1]
        scale_y = target_height / original_size[0]
        resized_bbox = [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_x),
            int(bbox[2] * scale_y),
            int(bbox[3] * scale_y),
        ]
        resized_boxes.append(resized_bbox)

    return {"x": np.stack(resized_arrays, axis=0), "y": np.array(resized_boxes)}
