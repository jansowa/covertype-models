from app.tools.file_helper import load_random_forest
from app.models.abstract_model import AbstractModel
from numpy.typing import ArrayLike


class RandomForest(AbstractModel):
    def __init__(self):
        self._classifier = load_random_forest()

    def predict(self, X) -> ArrayLike:
        return self._classifier.predict(X)
