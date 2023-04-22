import pandas as pd
from app.models.abstract_model import AbstractModel

class Heuristic(AbstractModel):
    def predict(self, X):
        X = pd.DataFrame(X)
        if X.shape[1] == 1:
            return Heuristic.__map_elevation(X.iloc[0, 0])
        return X.iloc[:, 0].apply(Heuristic.__map_elevation)

    @staticmethod
    def __map_elevation(elevation):
        if elevation < 2490:
            return 3
        if elevation > 3065:
            return 1
        return 2
