from app.tools.file_connector import load_random_forest


class RandomForest:
    __classifier = None

    def __init__(self):
        self.__classifier = load_random_forest()

    def predict(self, X):
        return self.__classifier.predict(X)
