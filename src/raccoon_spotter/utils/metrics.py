import numpy as np
from keras.losses import cosine_similarity


def mse(X, Y):
    return np.mean((X - Y) ** 2) / np.var(Y)


def cos(X, Y):
    return np.mean(cosine_similarity(X, Y))


def iou(X, Y):
    """
    Compute the average Intersection over Union (IoU) between corresponding regions in two sets of vectors.

    Parameters:
    - X (numpy.ndarray): The first set of vectors. Each row represents a region.
    - Y (numpy.ndarray): The second set of vectors. Must have the same number of rows as X.

    Returns:
    - float: The average IoU between corresponding regions in the sets of vectors.
    """
    # Make sure X and Y have the same number of rows
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Number of vectors must be the same in both sets")

    num_regions = X.shape[0]
    iou_values = []

    # Calculate IoU for each pair of vectors
    for i in range(num_regions):
        region1 = X[i]
        region2 = Y[i]

        intersection = np.sum(np.minimum(region1, region2))
        union = np.sum(np.maximum(region1, region2))

        if union == 0:
            iou = 0  # Define IoU as 0 if union is 0 to avoid division by zero
        else:
            iou = intersection / union

        iou_values.append(iou)

    # Calculate the average IoU
    return np.mean(iou_values)
