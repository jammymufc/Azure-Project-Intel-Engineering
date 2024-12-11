import unittest
import pandas as pd
import numpy as np
from model import loadData, splitData, buildModel, assessModel
from sklearn.ensemble import RandomForestClassifier

class TestTrain(unittest.TestCase):

    def setUp(self):
        # Mock data to use for tests
        self.testX = [
            [-6.544754, -2.9693909, 6.880844],
            [2.7863007, -9.415436, -0.44117737],
            [-7.437393, -2.8646545, 6.2667084],
            [-6.4114532, -3.5525818, 0.8227997],
            [-5.4878693, -2.1267395, 6.4857025],
            [-9.360733, -3.1264954, 0.62760925],
        ]
        self.testY = ['bike', 'sit', 'bike', 'walk', 'bike', 'stand']

    def test_loadData(self):
        # Test if loadData correctly processes the dataset
        X, Y = loadData('extracted_data_no_stairs.csv')
        # Ensure data has correct dimensions
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(Y))
        self.assertEqual(X.shape[1], 3)  # 3 features: x, y, z

    def test_splitData(self):
        # Test if splitData correctly splits the dataset
        X_train, X_test, Y_train, Y_test = splitData(self.testX, self.testY, test_size=0.2)
        self.assertEqual(len(X_train) + len(X_test), len(self.testX))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.testY))

    def test_buildModel(self):
        # Test if buildModel returns a RandomForestClassifier
        model = buildModel(self.testX, self.testY)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_assessModel(self):
        # Test if assessModel computes accuracy correctly
        model = buildModel(self.testX, self.testY)
        acc, predictions = assessModel(model, self.testX, self.testY)
        # Check if accuracy is a float in range [0, 1]
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        # Ensure predictions match input data size
        self.assertEqual(len(predictions), len(self.testY))

if __name__ == '__main__':
    unittest.main()
