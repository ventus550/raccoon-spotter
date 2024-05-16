import numpy as np
from sklearn.model_selection import train_test_split

from raccoon_spotter.utils.metrics import cos, iou, mse


def split_data(features_data_arr: np.ndarray, split_dataset: dict):
    X = features_data_arr["x"]
    y = features_data_arr["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_dataset["test_size"],
        random_state=split_dataset["random_state"],
        shuffle=split_dataset["shuffle"],
    )
    return X_test, y_test


def evaluate_model(trained_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = trained_model.predict(X_test)
    mse_score = mse(y_test, y_pred)
    cos_score = cos(y_test, y_pred)
    iou_score = iou(y_test, y_pred)
    metrics = {"mse": mse_score, "cosine_similarity": cos_score, "iou": iou_score}
    return metrics
