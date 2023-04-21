import pandas as pd
from app.models.abstractmodel import AbstractModel

class Heuristic(AbstractModel):
    def predict(self, X):
        X = pd.DataFrame(X)
        if X.shape[1] == 1:
            return Heuristic.map_elevation__(X.iloc[0, 0])
        return X.iloc[:, 0].apply(Heuristic.map_elevation__)

    @staticmethod
    def map_elevation__(elevation):
        if elevation < 2490:
            return 3
        if elevation > 3065:
            return 1
        return 2
