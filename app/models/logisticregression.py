from app.tools.file_connector import load_logistic_regression


class LogisticRegression:
    __classifier = None

    def __init__(self):
        self.__classifier = load_logistic_regression()

    def predict(self, X):
        return self.__classifier.predict(X)
