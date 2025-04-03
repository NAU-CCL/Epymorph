from unittest.mock import MagicMock

import numpy as np
import pytest

from epymorph.adrio import acs5
from epymorph.adrio.adrio import ADRIOContextError
from epymorph.adrio.adrio import ADRIOPrototype as ADRIO  # noqa: N814
from epymorph.data_shape import DataShape, DataShapeMatcher, Dimensions
from epymorph.data_type import dtype_as_np
from epymorph.geography.scope import GeoScope
from epymorph.kit import *
from epymorph.simulation import Context
from epymorph.util import NumpyTypeError, check_ndarray, match

# To re-record this test:
# load your census API key into the environment, then
# uv run pytest tests/slow/adrio/acs5_test.py --record-mode=rewrite
pytestmark = [pytest.mark.default_cassette("acs5.yaml")]


def drop_resp_headers(response):
    # we don't need the headers to be saved
    # and they might contain sensitive info
    response["headers"] = {}
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_query_parameters": ["key"],
        "before_record_response": drop_resp_headers,
    }


# List all of the attributes we want to test, values are tuples containing:
# - the ADRIO that fetches it
# - the expected type of the result
# - the expected shape of the result
_attributes_to_test: dict[str, tuple[ADRIO, type[int | float], DataShape]] = {
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
        int,
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
        (),
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
        ("dissimilarity_index", "gini_index"),
    ),
]


@pytest.mark.parametrize(("time_frame", "scope", "skip"), _test_parameters)
def test_attributes(time_frame: TimeFrame, scope: GeoScope, skip: tuple[str, ...]):
    # TODO: replace this test...
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


##############
# _FetchACS5 #
##############


def test_fetch_acs5_validate_context(monkeypatch):
    class MockADRIO(acs5._FetchACS5):
        @property
        def _variables(self):
            raise NotImplementedError()

        @property
        def result_format(self):
            raise NotImplementedError()

    adrio = MockADRIO()
    states = StateScope.in_states(["AZ", "NM"], year=2020)
    counties = states.lower_granularity()
    tracts = counties.lower_granularity()
    cbgs = tracts.lower_granularity()

    # Invalid if we don't have a Census key
    monkeypatch.delenv("API_KEY__census.gov", raising=False)
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    with pytest.raises(ADRIOContextError, match="Census API key is required"):
        adrio.validate_context(Context.of(scope=states))

    monkeypatch.setenv("CENSUS_API_KEY", "abcd1234")

    # Valid contexts:
    adrio.validate_context(Context.of(scope=states))
    adrio.validate_context(Context.of(scope=counties))
    adrio.validate_context(Context.of(scope=tracts))
    adrio.validate_context(Context.of(scope=cbgs))

    # Invalid contexts:
    with pytest.raises(ADRIOContextError, match="US Census geo scope required"):
        adrio.validate_context(Context.of(scope=CustomScope(["A", "B", "C"])))

    states_2008 = MagicMock(spec=StateScope)
    states_2008.year = 2008  # mock b/c this isn't a valid scope in the first place
    with pytest.raises(ADRIOContextError, match="not a supported year for ACS5 data"):
        adrio.validate_context(Context.of(scope=states_2008))

    states_2024 = MagicMock(spec=StateScope)
    states_2024.year = 2024  # mock b/c this isn't a valid scope in the first place
    with pytest.raises(ADRIOContextError, match="not a supported year for ACS5 data"):
        adrio.validate_context(Context.of(scope=states_2024))


##############
# Population #
##############


@pytest.mark.vcr
def test_population_state():
    actual = (
        acs5.Population()
        .with_context(
            scope=CountyScope.in_states(["AZ"], year=2021),
        )
        .evaluate()
    )
    # fmt: off
    expected = np.array([
        66473, 125092, 144942, 53211, 38145,
        9542, 16845, 4367186, 211274, 106609,
        1035063, 420625, 47463, 233789, 202944,
    ], dtype=np.int64)
    # fmt: on
    np.testing.assert_array_equal(actual, expected, strict=True)
