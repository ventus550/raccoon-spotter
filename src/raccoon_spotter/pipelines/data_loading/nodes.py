from typing import Dict

import numpy as np
import pandas as pd


def _structure_data(raccoon_images=Dict[str, np.ndarray], raccoon_labels=pd.DataFrame):
    bounding_box = ["xmin", "xmax", "ymin", "ymax"]
    return [
        (
            np.array(raccoon_images[f"images/{row.filename}"]),
            np.array(row[bounding_box]),
        )
        for _, row in raccoon_labels.iterrows()
    ]


def construct_data_array(
    raccoon_images=Dict[str, np.ndarray], raccoon_labels=pd.DataFrame
):
    z = zip(*_structure_data(raccoon_images, raccoon_labels))
    x, y = tuple(z)
    return dict(x=np.array(x, dtype=object), y=np.array(y))
