from enum import Enum

from app.models.abstract_model import AbstractModel
from app.models.heuristic import Heuristic
from app.models.logistic_regression import LogisticRegression
from app.models.neural_network import NeuralNetwork
from app.models.random_forest import RandomForest


class ModelTypeEnum(str, Enum):
    heuristic = "heuristic"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    neural_network = "neural_network"


def get_model_type(model_type: ModelTypeEnum) -> AbstractModel:
    if model_type == ModelTypeEnum.heuristic:
        return Heuristic()
    elif model_type == ModelTypeEnum.random_forest:
        return RandomForest()
    elif model_type == ModelTypeEnum.logistic_regression:
        return LogisticRegression()
    elif model_type == ModelTypeEnum.neural_network:
        return NeuralNetwork()
    else:
        raise NotImplementedError
