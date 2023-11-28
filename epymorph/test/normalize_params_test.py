import unittest

import numpy as np

from epymorph.context import normalize_params
from epymorph.data_shape import Shapes
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, NO_DURATION, AttribDef, CentroidDType,
                               StaticGeoSpec)
from epymorph.geo.static import StaticGeo


class TestNormalizeParams(unittest.TestCase):

    @staticmethod
    def TestGeo() -> Geo:
        """Load the single_pop geo."""
        spec = StaticGeoSpec(
            attributes=[
                LABEL,
                AttribDef('geoid', np.str_, Shapes.N),
                AttribDef('centroid', CentroidDType, Shapes.N),
                AttribDef('population', np.int64, Shapes.N),
                AttribDef('commuters', np.int64, Shapes.NxN),
            ],
            time_period=NO_DURATION
        )
        return StaticGeo(spec, {
            'label': np.array(['AZ'], dtype=np.str_),
            'geoid': np.array(['04'], dtype=np.str_),
            'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
            'population': np.array([100_000], dtype=np.int64),
            'commuters': np.array([[0]], dtype=np.int64)
        })

    def setUp(self) -> None:
        self.testgeo = self.TestGeo()

    def test_normalize_params_with_empty_data(self):
        """
        Test that the function normalizes an empty data dictionary.
        """

        normalized_params = normalize_params({}, self.testgeo, 100, {})

        self.assertEqual(normalized_params, {})

    def test_normalize_params_with_simple_values(self):
        """
        Test that the function normalizes simple values to numpy arrays.
        """

        data = {"a": 1, "b": 2.0, "c": "3"}

        normalized_params = normalize_params(
            data, self.testgeo, 100, {})

        self.assertEqual(normalized_params["a"], np.array(1))
        self.assertEqual(normalized_params["b"], np.array(2.0))
        self.assertEqual(normalized_params["c"], np.array("3"))

    def test_normalize_params_with_lists(self):
        """
        Test that the function normalizes lists to numpy arrays.
        """

        data = {"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": ["5", "6", "7"]}

        normalized_params = normalize_params(
            data, self.testgeo, 100, {})

        self.assertTrue(np.array_equal(normalized_params["a"], np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(
            normalized_params["b"], np.array([2.0, 3.0, 4.0])))
        self.assertTrue(np.array_equal(
            normalized_params["c"], np.array(["5", "6", "7"])))

    def test_normalize_params_with_functions(self):
        """
        Test that the function normalizes functions to numpy arrays.
        """

        def my_function(t, _):
            return t ** 2

        data = {"a": my_function}

        normalized_params = normalize_params(
            data, self.testgeo, 100, {})

        self.assertTrue(np.array_equal(normalized_params["a"], np.array(
            [my_function(t, '_') for t in range(100)])))

    def test_normalize_params_with_function_strings(self):
        """
        Test that the function normalizes function strings to numpy arrays.
        """
        def my_function(t, _):
            return t ** 2

        data = {
            "a": '''def cal(t,_):
        return t ** 2'''
        }

        normalized_params = normalize_params(
            data, self.testgeo, 100, {})

        self.assertTrue(np.array_equal(normalized_params["a"], np.array(
            [my_function(t, None) for t in range(100)])))


if __name__ == '__main__':
    unittest.main()
