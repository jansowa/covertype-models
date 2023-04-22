class AbstractModel(object):
    _classifier = None

    def predict(self, X):
        return [2 for _ in range(X.shape[0])]  # the most frequent value
