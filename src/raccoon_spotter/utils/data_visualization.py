import cv2
import numpy as np
from kedro.config import OmegaConfigLoader


def verify_bounding_box(index):
    catalog_conf = OmegaConfigLoader(conf_source=".")["catalog"]
    path = catalog_conf["raccoon_data_array"]["filepath"]
    data_arr = np.load(path, allow_pickle=True)
    image_array = data_arr["x"][index]
    bounding_box = data_arr["y"][index]
    img = cv2.rectangle(
        image_array,
        (bounding_box[0], bounding_box[2]),
        (bounding_box[1], bounding_box[3]),
        (255, 0, 0),
        2,
    )
    return img


def draw_bounding_box(image_array: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    Parameters:
    - img (np.ndarray): The image array.
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
