import cv2
import numpy as np


def draw_bounding_box(image_array: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    Parameters:
    - image_array (np.ndarray): The image array.
    - bounding_boxes (np.ndarray): xmin, xmax, ymin, ymax.

    Returns:
    - img_with_boxes (np.ndarray): The image array with the bounding boxes drawn.
    """
    img = image_array.copy()
    img_with_boxes = cv2.rectangle(
        img,
        (bounding_box[0], bounding_box[2]),
        (bounding_box[1], bounding_box[3]),
        (255, 0, 0),
        2,
    )
    return img_with_boxes
