import numpy as np

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.params import ParamFunction


class ExponentialTransform(ParamFunction[np.float64]):
    @property
    def requirements(self):
        if self._value_req is not None:
            return (self._value_req,)
        else:
            return ()

    def __init__(self, other: str):
        self._value_req = AttributeDef(other, float, Shapes.TxN)

    def evaluate(self):
        return np.exp(self.data(self._value_req.name))


class ShiftTransform(ParamFunction[np.float64]):
    @property
    def requirements(self):
        return (self._first_req, self._second_req)

    def __init__(self, first: str, second: str):
        self._first_req = AttributeDef(first, float, Shapes.TxN)
        self._second_req = AttributeDef(second, float, Shapes.TxN)

    def evaluate(self):
        return self.data(self._first_req.name) + self.data(self._second_req.name)
