from enum import Enum


class ModelTypeEnum(str, Enum):
    heuristic = "heuristic"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    neural_network = "neural_network"
