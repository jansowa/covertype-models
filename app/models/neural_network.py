from app.tools.file_helper import load_neural_network, load_min_max_scaler
from app.models import AbstractModel
from app.tools.evaluation import predict_proba_to_class
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import ArrayLike


class NeuralNetwork(AbstractModel):
    __scaler: MinMaxScaler

    def __init__(self):
        self._classifier = load_neural_network()
        self.__scaler = load_min_max_scaler()

    def predict(self, X) -> ArrayLike:
        X_scaled = self.__scaler.transform(X)
        return predict_proba_to_class(self._classifier.predict(X_scaled))
