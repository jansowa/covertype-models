from app.tools.file_connector import load_neural_network
from app.models.abstract_model import AbstractModel
from app.tools.evaluation import predict_proba_to_class


class NeuralNetwork(AbstractModel):

    def __init__(self):
        self._classifier = load_neural_network()

    def predict(self, X):
        return predict_proba_to_class(self._classifier.predict(X))