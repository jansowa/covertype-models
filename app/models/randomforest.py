from app.tools.file_connector import load_random_forest
from app.models.abstractmodel import AbstractModel

class RandomForest(AbstractModel):
    def __init__(self):
        self._classifier = load_random_forest()

    def predict(self, X):
        return self._classifier.predict(X)
