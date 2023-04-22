import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVED_MODELS_PATH = ROOT_DIR + "/models/saved"

RANDOM_FOREST_PATH = SAVED_MODELS_PATH + "/random-forest.joblib"
LOGISTIC_REGRESSION_PATH = SAVED_MODELS_PATH + "/logistic-regression.joblib"
NEURAL_NETWORK_PATH = SAVED_MODELS_PATH + "/neural-network.ckpt"
NEURAL_NETWORK_PARAMS_PATH = SAVED_MODELS_PATH + "/best_nn_params.csv"
MIN_MAX_SCALER_PATH = SAVED_MODELS_PATH + "/min_max_scaler.joblib"

DATASET_PATH = ROOT_DIR + "/../covtype.data"
