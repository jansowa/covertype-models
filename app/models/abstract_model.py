class AbstractModel(object):
    _classifier = None

    def predict(self, X):
        return 2 # the most frequent value