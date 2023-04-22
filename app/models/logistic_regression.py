from app.tools.file_helper import load_logistic_regression
from app.models import AbstractModel
from numpy.typing import ArrayLike


class LogisticRegression(AbstractModel):

    def __init__(self):
        self._classifier = load_logistic_regression()

    def predict(self, X) -> ArrayLike:
        return self._classifier.predict(X)
