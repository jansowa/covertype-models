from joblib import dump, load
from app.file_paths import RANDOM_FOREST_PATH, LOGISTIC_REGRESSION_PATH, NEURAL_NETWORK_PATH, MIN_MAX_SCALER_PATH


def save_model(model, path: str) -> None:
    dump(model, path, 9)


def load_model(path: str):
    return load(path)


def load_random_forest():
    return load_model(RANDOM_FOREST_PATH)


def load_logistic_regression():
    return load_model(LOGISTIC_REGRESSION_PATH)


def load_neural_network_weights(neural_network):
    neural_network.load_weights(NEURAL_NETWORK_PATH)
    return neural_network


def load_min_max_scaler():
    return load_model(MIN_MAX_SCALER_PATH)
