from itertools import chain

import cv2
import mplcyberpunk
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

def _dim_image(image, dim_factor):
    return cv2.convertScaleAbs(image, alpha=dim_factor, beta=0)

# Function to superimpose a region on an image
def _superimpose_region(background, region, minx, maxx, miny, maxy):
    # Superimpose the region on the background
    background[miny:maxy, minx:maxx] = region

    # Draw a border around the region
    cv2.rectangle(background, (minx, miny), (maxx, maxy), (0, 255, 0), 1)
    return background 

def roi(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Highlight the region of intereset of the image.

    Parameters:
    - image (np.ndarray): The image array.
    - box (np.ndarray): xmin, xmax, ymin, ymax.

    Returns:
    - roi (np.ndarray): The image array with the region of intereset highlighted.
    """
    xmin, xmax, ymin, ymax = box
    dimmed_image = _dim_image(image, dim_factor=0.5)
    region = image[ymin:ymax, xmin:xmax]
    return np.array(_superimpose_region(dimmed_image, region, xmin, xmax, ymin, ymax))

def radialplot(categories, data, intervals=8):
    """
    Create a radial multiplot.

    Parameters:
    - categories (list): A list of categories to be displayed on the radial plot of the same shape as values.
    - data (dict or list): A dictionary mapping categories to lists of numerical values or a list of numerical values. If a dictionary is provided, each key-value pair represents a category and its corresponding values. If a list is provided, it will be treated as a single category.

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib Figure object containing the radial plot.

    Example:
    ```
    data = {"Data1": [10, 15, 20, 25], "Data2": [5, 10, 15, 20]}
    categories = ["A", "B", "C", "D"]
    radialplot(data, categories)
    plt.show()
    ```
    """

    # Set up the data so that we can close the 'loop' of the area
    data = data.copy()
    if not isinstance(data, dict):
        data = {"unknown": data}

    # Check validity
    assert np.array(list(data.values())).shape[1] == len(categories)

    categories = [*categories, categories[0]]
    for k, v in data.items():
        data[k] = [*v, v[0]]

    # Set up the label potisions around the circle circumference
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))

    with plt.style.context("cyberpunk"):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Add our data as separate axes
        for values in data.values():
            ax.plot(label_loc, values, lw=2)
            ax.fill(label_loc, values, alpha=0.3)

        # Convert the lines and labels to a polar grid
        plt.thetagrids(np.degrees(label_loc), labels=categories)

        # Set up the grid and line properties
        ax.tick_params(axis="both", which="major", pad=30, labelsize=10)

        ax.spines["polar"].set_linewidth(3)
        ax.spines["polar"].set_color((1, 1, 1, 0.1))
        ax.grid(color="white", alpha=0.3)
        fig.tight_layout(pad=0)

        # Setup the radial lines
        max_value = max(chain(*data.values()))
        ticks = np.linspace(0, max_value, intervals)
        ax.set_ylim(0, max_value)
        ax.set_yticks(ticks)

        # Add a legend with custom position and handles
        ax.legend(
            handles=[
                Patch(facecolor=f"C{i}", alpha=0.5, label=label)
                for i, label in enumerate(data)
            ],
            bbox_to_anchor=(1.3, 0.2),
            fontsize=16,
            frameon=False,
        )

        mplcyberpunk.add_glow_effects()
        return fig


def kdeplot(sample=np.random.logistic(0, 1, 1000)):
    with plt.style.context("cyberpunk"):
        sns.kdeplot(sample)
        mplcyberpunk.add_glow_effects()
