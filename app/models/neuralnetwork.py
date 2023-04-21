from app.tools.file_connector import load_neural_network
from app.models.abstractmodel import AbstractModel

class NeuralNetwork(AbstractModel):

    def __init__(self):
        self._classifier = load_neural_network()

    def predict(self, X):
        return self._classifier.predict(X)