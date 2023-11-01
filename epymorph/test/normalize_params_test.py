import unittest
import numpy as np

from epymorph.context import normalize_params

class TestNormalizeParams(unittest.TestCase):

    def test_normalize_params_with_empty_data(self):
        """
        Test that the function normalizes an empty data dictionary.
        """

        data = {}
        compartments = 10
        duration = 100
        dtypes = {}

        normalized_params = normalize_params(data, compartments, duration, dtypes)

        self.assertEqual(normalized_params, {})

    def test_normalize_params_with_simple_values(self):
        """
        Test that the function normalizes simple values to numpy arrays.
        """

        data = {"a": 1, "b": 2.0, "c": "3"}
        compartments = 10
        duration = 100
        dtypes = {}

        normalized_params = normalize_params(data, compartments, duration, dtypes)

        self.assertEqual(normalized_params["a"], np.array(1))
        self.assertEqual(normalized_params["b"], np.array(2.0))
        self.assertEqual(normalized_params["c"], np.array("3"))

    def test_normalize_params_with_lists(self):
        """
        Test that the function normalizes lists to numpy arrays.
        """

        data = {"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": ["5", "6", "7"]}
        compartments = 10
        duration = 100
        dtypes = {}

        normalized_params = normalize_params(data, compartments, duration, dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(normalized_params["b"], np.array([2.0, 3.0, 4.0])))
        self.assertTrue(np.array_equal(normalized_params["c"], np.array(["5", "6", "7"])))


    def test_normalize_params_with_functions(self):
        """
        Test that the function normalizes functions to numpy arrays.
        """

        def my_function(t):
            return t ** 2

        data = {"a": my_function}
        compartments = 10
        duration = 100
        dtypes = {}

        normalized_params = normalize_params(data, compartments, duration, dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array([my_function(t) for t in range(duration)])))

    def test_normalize_params_with_function_strings(self):
        """
        Test that the function normalizes function strings to numpy arrays.
        """
        def my_function(t):
            return t ** 2

        data = {
       "a": '''def cal(t):
        return t ** 2'''
        }   
        compartments = 10
        duration = 100
        dtypes = {}

        normalized_params = normalize_params(data, compartments, duration, dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array([my_function(t) for t in range(duration)])))
