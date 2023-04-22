import unittest

import numpy.testing

from app.tools.evaluation import predict_proba_to_class


class EvaluationTest(unittest.TestCase):

    def test_predict_proba_to_class(self):
        proba = [[0.1, 0.2, 0.5, 0.0, 0.1, 0, 0.1],
                 [0, 0, 0, 0, 0, 0, 1]]
        numpy.testing.assert_array_equal(
            predict_proba_to_class(proba),
            [3, 7]
        )


if __name__ == '__main__':
    unittest.main()
