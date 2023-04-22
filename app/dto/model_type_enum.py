from enum import Enum

from app.models import AbstractModel, Heuristic, LogisticRegression, NeuralNetwork, RandomForest


class ModelTypeEnum(str, Enum):
    heuristic = "heuristic"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    neural_network = "neural_network"

    def get_model_by_type(self) -> AbstractModel:
        if self == ModelTypeEnum.heuristic:
            return Heuristic()
        elif self == ModelTypeEnum.random_forest:
            return RandomForest()
        elif self == ModelTypeEnum.logistic_regression:
            return LogisticRegression()
        elif self == ModelTypeEnum.neural_network:
            return NeuralNetwork()
        else:
            raise NotImplementedError
