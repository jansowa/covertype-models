from app.tools.file_connector import load_logistic_regression
from app.models.abstractmodel import AbstractModel

class LogisticRegression(AbstractModel):

    def __init__(self):
        self._classifier = load_logistic_regression()

    def predict(self, X):
        return self._classifier.predict(X)
