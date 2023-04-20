# TODO: change name of this file

from joblib import dump, load
from app.definitions import ROOT_DIR

SAVED_MODELS_PATH = ROOT_DIR + "/models/saved"
RANDOM_FOREST_PATH = SAVED_MODELS_PATH + "/random-forest.joblib"
LOGISTIC_REGRESSION_PATH = SAVED_MODELS_PATH + "/logistic-regression.joblib"


def save_model(model, path: str):
    dump(model, path)

def load_model(path: str):
    return load(path)

def load_random_forest():
    return load_model(RANDOM_FOREST_PATH)

def load_logistic_regression():
    return load_model(LOGISTIC_REGRESSION_PATH)