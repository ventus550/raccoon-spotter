import matplotlib.pyplot as plt
import numpy as np
from keras import Model

from raccoon_spotter.utils.data_visualization import radialplot, roi
from raccoon_spotter.utils.metrics import cos, iou, mse
from raccoon_spotter.utils.wandb import Client


def evaluate_model(trained_model: Model, testing_data_arrays: np.ndarray) -> dict:
    X, Y = testing_data_arrays.values()
    P = trained_model.predict(X)
    metrics = {"mse": mse(Y, P), "cos": cos(Y, P), "iou": iou(Y, P)}
    Client().log(metrics)
    return metrics


def radialplot_comparison(
    trained_model: Model,
    untrained_model: Model,
    testing_data_arrays: np.ndarray,
) -> dict:
    trained_model_metrics = evaluate_model(trained_model, testing_data_arrays)
    untrained_model_metrics = evaluate_model(untrained_model, testing_data_arrays)

    mse1, cos1, iou1 = trained_model_metrics.values()
    mse0, cos0, iou0 = untrained_model_metrics.values()

    def scaled(array):
        return array / max(array)

    mse0, mse1 = scaled(np.array([mse0, mse1]))
    cos0, cos1 = scaled(np.array([1 + cos0, 1 + cos1]))
    iou0, iou1 = scaled(np.array([1 - iou0, 1 - iou1]))

    return radialplot(
        [
            "\nMean\nSquared\nError",
            "\nCosine\nSimilarity",
            "Intersection\nOver\nUnion\nComplement",
        ],
        dict(
            untrained_model=[mse0, cos0, iou0],
            trained_model=[mse1, cos1, iou1],
        ),
        intervals=9,
    )


def sample_model(model: Model, training_data_arrays: np.ndarray):
    X, Y = training_data_arrays.values()
    P = model.predict(X).astype(int)
    assert len(P) >= 6  # noqa: PLR2004
    fig, axs = plt.subplots(6, 2, figsize=(12, 24))
    for i, ax in enumerate(axs):
        ax[0].imshow(roi(X[i], Y[i]))
        ax[1].imshow(roi(X[i], P[i]))
    return fig
