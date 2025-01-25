import pytest

from epymorph.adrio import acs5, commuting_flows
from epymorph.adrio.adrio import Adrio
from epymorph.data_shape import DataShape, DataShapeMatcher, Dimensions
from epymorph.data_type import dtype_as_np
from epymorph.geography.scope import GeoScope
from epymorph.kit import *
from epymorph.util import NumpyTypeError, check_ndarray, match

# List all of the attributes we want to test, values are tuples containing:
# - the ADRIO that fetches it
# - the expected type of the result
# - the expected shape of the result
_attributes_to_test: dict[str, tuple[Adrio, type[int | float], DataShape]] = {
    "population": (
        acs5.Population(),
        int,
        Shapes.N,
    ),
    "population_by_age_table": (
        acs5.PopulationByAgeTable(),
        int,
        Shapes.NxA,
    ),
    "population_by_age": (
        acs5.PopulationByAge(18, 24),
        int,
        Shapes.N,
    ),
    "average_household_size": (
        acs5.AverageHouseholdSize(),
        float,
        Shapes.N,
    ),
    "dissimilarity_index": (
        acs5.DissimilarityIndex("White", "Black"),
        float,
        Shapes.N,
    ),
    "commuters": (
        commuting_flows.Commuters(),
        int,
        Shapes.NxN,
    ),
    "gini_index": (
        acs5.GiniIndex(),
        float,
        Shapes.N,
    ),
    "median_age": (
        acs5.MedianAge(),
        float,
        Shapes.N,
    ),
    "median_income": (
        acs5.MedianIncome(),
        float,
        Shapes.N,
    ),
}

_test_parameters = [
    # Test 0: state scope
    (
        TimeFrame.year(2020),
        StateScope.in_states(["NY", "NJ", "MD", "VA"], year=2020),
        (),
    ),
    # Test 1: another state scope
    (
        TimeFrame.year(2020),
        StateScope.in_states(["04", "08"], year=2020),
        (),
    ),
    # Test 2: county scope
    (
        TimeFrame.year(2020),
        CountyScope.in_counties(["35001", "04013", "04017"], year=2020),
        (),
    ),
    # Test 3: tract scope
    (
        TimeFrame.year(2020),
        TractScope.in_tracts(
            ["35001000720", "35001000904", "35001000906", "04027011405", "04027011407"],
            year=2020,
        ),
        ("commuters",),
    ),
    # Test 4: CBG scope
    (
        TimeFrame.year(2020),
        BlockGroupScope.in_block_groups(
            [
                "350010007201",
                "350010009041",
                "350010009061",
                "040270114053",
                "040270114072",
            ],
            year=2020,
        ),
        ("commuters", "dissimilarity_index"),
    ),
]


@pytest.mark.parametrize(("time_frame", "scope", "skip"), _test_parameters)
def test_attributes(time_frame: TimeFrame, scope: GeoScope, skip: tuple[str, ...]):
    dim = Dimensions.of(
        T=time_frame.duration_days,
        N=scope.nodes,
    )
    params = {k: v[0] for k, v in _attributes_to_test.items()}
    for name, (adrio, dtype, shape) in _attributes_to_test.items():
        if name in skip:
            continue
        try:
            actual = adrio.with_context(
                params=params,
                scope=scope,
                time_frame=time_frame,
            ).evaluate()
            check_ndarray(
                actual,
                dtype=match.dtype(dtype_as_np(dtype)),
                shape=DataShapeMatcher(shape, dim, exact=True),
            )
        except NumpyTypeError as e:
            pytest.fail(f"attribute '{name}': {e}")
