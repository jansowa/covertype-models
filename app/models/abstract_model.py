from numpy.typing import ArrayLike


class AbstractModel(object):
    _classifier = None

    def predict(self, X) -> ArrayLike:
        return [2 for _ in range(X.shape[0])]  # the most frequent value
