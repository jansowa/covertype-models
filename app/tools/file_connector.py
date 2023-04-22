# TODO: change name of this file

from joblib import dump, load
from app.definitions import ROOT_DIR
from app.models.untrained_model import untrained_neural_network

SAVED_MODELS_PATH = ROOT_DIR + "/models/saved"
RANDOM_FOREST_PATH = SAVED_MODELS_PATH + "/random-forest.joblib"
LOGISTIC_REGRESSION_PATH = SAVED_MODELS_PATH + "/logistic-regression.joblib"
NEURAL_NETWORK_PATH = SAVED_MODELS_PATH + "/neural-network.ckpt"
NEURAL_NETWORK_PARAMS_PATH = SAVED_MODELS_PATH + "/best_nn_params.csv"
MIN_MAX_SCALER_PATH = SAVED_MODELS_PATH + "/min_max_scaler.joblib"


def save_model(model, path: str):
    dump(model, path, 9)


def load_model(path: str):
    return load(path)


def load_random_forest():
    return load_model(RANDOM_FOREST_PATH)


def load_logistic_regression():
    return load_model(LOGISTIC_REGRESSION_PATH)


def load_neural_network():
    neural_network = untrained_neural_network()
    neural_network.load_weights(NEURAL_NETWORK_PATH)
    return neural_network


def load_min_max_scaler():
    return load_model(MIN_MAX_SCALER_PATH)
