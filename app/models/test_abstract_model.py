import unittest
from random import random
import pandas as pd
from app.models import AbstractModel
import numpy.testing


class AbstractModelTest(unittest.TestCase):
    def test_predict(self):
        samples_number = 25
        X = pd.DataFrame([[random() for _ in range(54)] for _ in range(samples_number)])
        model = AbstractModel()
        numpy.testing.assert_array_equal(
            model.predict(X), [2] * samples_number)


if __name__ == '__main__':
    unittest.main()
