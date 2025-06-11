import numpy as np
import pytest
from typing_extensions import override

from epymorph.adrio import adrio
from epymorph.adrio.validation import ResultFormat
from epymorph.data_shape import Shapes
from epymorph.geography.custom import CustomScope
from epymorph.geography.us_census import CountyScope
from epymorph.simulation import Context


def test_node_id_01():
    expected = np.array(["04005", "04013"], dtype=np.str_)
    actual = (
        adrio.NodeID()
        .with_context(
            scope=CountyScope.in_counties(["04005", "04013"], year=2020),
        )
        .evaluate()
    )
    assert np.array_equal(actual, expected)

    expected = np.array(["A", "B", "C"], dtype=np.str_)
    actual = (
        adrio.NodeID()
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )
    assert np.array_equal(actual, expected)


def test_node_id_02():
    # Error: no scope provided
    with pytest.raises(adrio.ADRIOContextError):
        adrio.NodeID().with_context().evaluate()


class _TestADRIO(adrio.ADRIO[np.float64, np.float64]):
    # For ease of testing, we need an ADRIO that just returns the values it was given.

    def __init__(self, values):
        self.values = values

    @property
    @override
    def result_format(self):
        return ResultFormat(shape=Shapes.N, dtype=np.float64)

    @override
    def validate_context(self, context: Context) -> None:
        pass

    @override
    def inspect(self) -> adrio.InspectResult[np.float64, np.float64]:
        return adrio.InspectResult(
            adrio=self,
            source=self.values,
            result=self.values,
            shape=Shapes.N,
            dtype=np.float64,
            issues={},
        )


def test_scale():
    expected = np.array([110, 220, 330], dtype=np.float64)
    actual = (
        adrio.Scale(
            parent=_TestADRIO(np.array([10, 20, 30], dtype=np.float64)),
            factor=11,
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )
    assert np.array_equal(actual, expected)


def test_population_per_km2():
    pop = _TestADRIO(np.array([10, 20, 30], dtype=np.int64))
    area = _TestADRIO(np.array([5, 4, 3], dtype=np.float64))
    expected = np.array([2, 5, 10], dtype=np.float64)
    actual = (
        adrio.PopulationPerKM2()
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
            params={
                "population": pop,
                "land_area_km2": area,
            },
        )
        .evaluate()
    )
    assert np.array_equal(actual, expected)
