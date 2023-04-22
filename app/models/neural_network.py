from app.tools.file_connector import load_neural_network
from app.models.abstract_model import AbstractModel
from app.tools.evaluation import predict_proba_to_class
from sklearn.preprocessing import MinMaxScaler


class NeuralNetwork(AbstractModel):

    def __init__(self):
        self._classifier = load_neural_network()

    def predict(self, X):
        # TODO: load this scaler from file trained on the whole dataset
        X_scaled = MinMaxScaler().fit_transform(X)
        return predict_proba_to_class(self._classifier.predict(X_scaled))
