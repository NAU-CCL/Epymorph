from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from epymorph.adrio import acs5
from epymorph.adrio.adrio import ADRIOContextError, ADRIOProcessingError, FetchADRIO
from epymorph.error import DataAttributeError
from epymorph.kit import *
from epymorph.simulation import Context

# NOTE: these tests use VCR to record HTTP requests.
# To re-record this test, load a census API key into the environment, then:
# uv run pytest tests/slow/adrio/acs5_test.py --vcr-mode=record


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_query_parameters": ["key"]}


##############
# _FetchACS5 #
##############


def test_fetch_acs5_validate_context(monkeypatch):
    class MockADRIO(acs5._ACS5FetchMixin, FetchADRIO):
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


def _test_population(scope, expected):
    actual = acs5.Population().with_context(scope=scope).evaluate()
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.vcr
def test_population_state():
    _test_population(
        scope=StateScope.in_states(["AZ", "CO", "NM", "UT"], year=2021),
        expected=np.array([7079203, 5723176, 2109366, 3231370], dtype=np.int64),
    )


@pytest.mark.vcr
def test_population_county():
    # fmt: off
    _test_population(
        scope=CountyScope.in_states(["AZ"], year=2020),
        expected=np.array([
            71714, 126442, 142254, 53846, 38304,
            9465, 21035, 4412779, 210998, 110271,
            1038476, 447559, 46594, 232396, 211931,
        ], dtype=np.int64),
    )
    # fmt: on


@pytest.mark.vcr
def test_population_tract():
    # fmt: off
    _test_population(
        scope=TractScope.in_counties(["Coconino, AZ"], year=2019),
        expected = np.array([
            3541, 3700, 6034, 5319, 4240,
            5378, 3986, 6522, 6587, 12877,
            4495, 8003, 2788, 5317, 5404,
            3006, 3140, 3712, 1872, 7531,
            6882, 6182, 3712, 3594, 4904,
            4164, 4063, 4321,
        ], dtype=np.int64),
    )
    # fmt: on


@pytest.mark.vcr
def test_population_cbg():
    # fmt: off
    _test_population(
        scope=BlockGroupScope.in_tracts(
            ["04005000100", "04005001000", "04005942201", "04005945100"],
            year=2018,
        ),
        expected = np.array([
            1375, 1010, 1132, 4711, 3654,
            4003, 1777, 801, 1217, 1419,
            969, 743, 812,
        ], dtype=np.int64),
    )
    # fmt: on


def test_population_invalid_scope():
    with pytest.raises(ADRIOContextError, match="US Census geo scope required"):
        acs5.Population().with_context(scope=CustomScope(["A", "B", "C"])).evaluate()


def test_population_invalid_year():
    # It's not really possible to construct a scope with an invalid year, so mock it.
    s = CountyScope.in_states(["AZ"], year=2020)
    scope = Mock(spec=s, wraps=s)
    scope.year = 1999  # Census API will not allow us to party like it's 1999.
    with pytest.raises(ADRIOContextError, match="not a supported year"):
        acs5.Population().with_context(scope=scope).evaluate()


@pytest.mark.vcr
def test_population_by_age_table():
    actual = (
        acs5.PopulationByAgeTable()
        .with_context(
            scope=CountyScope.in_counties(
                ["Maricopa, AZ", "Coconino, AZ", "Bernalillo, NM"],
                year=2020,
            ),
        )
        .evaluate()
    )
    # fmt: off
    expected = np.array([
        [142254, 70124, 3905, 3803, 4614, 2646, 4265, 2364, 2051, 4581,
         5805, 4524, 4128, 3757, 3639, 3597, 3829, 1690, 2327, 1405,
         1789, 2565, 1513, 871, 456, 72130, 3726, 3451, 4498, 2397,
         7673, 2771, 1910, 3285, 5243, 4190, 4110, 3568, 3575, 3777,
         4591, 1503, 2506, 1415, 2168, 2241, 1774, 878, 880],
        [4412779, 2181967, 141837, 145844, 156646, 91961, 59934, 30285,
         29928, 89834, 170254, 158532, 146078, 141371, 139718, 133277,
         127838, 50484, 65661, 39585, 58031, 81642, 56271, 36166, 30790,
         2230812, 135478, 139682, 151403, 88167, 56378, 28329, 29408, 85295,
         163997, 152910, 146207, 141140, 142546, 136798, 137603, 51557, 75303,
         47187, 69092, 93582, 66264, 43536, 48950],
        [679037, 332754, 19362, 20981, 21895, 13101, 8571, 4206, 4861, 13651,
         26135, 25566, 23580, 20397, 19952, 20019, 21550, 7861, 12077, 7543,
         9648, 13310, 8090, 5413, 4985, 346283, 18507, 18592, 22513, 12755,
         8225, 4792, 3946, 13140, 25116, 24773, 23402, 20431, 20544, 21204,
         24000, 9298, 13149, 8090, 11990, 15755, 9873, 7349, 8839],
    ], dtype=np.int64)
    # fmt: on
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.vcr
def test_population_by_age():
    context = {
        "scope": CountyScope.in_counties(
            ["Maricopa, AZ", "Coconino, AZ", "Bernalillo, NM"],
            year=2020,
        ),
        "params": {
            "population_by_age_table": acs5.PopulationByAgeTable(),
        },
    }

    a, b, c, d = 18, 19, 20, 21
    ages_ab = acs5.PopulationByAge(a, b).with_context(**context).evaluate()
    ages_cd = acs5.PopulationByAge(c, d).with_context(**context).evaluate()
    ages_ad = acs5.PopulationByAge(a, d).with_context(**context).evaluate()

    # Check one set of values...
    expected = np.array([11938, 116312, 16796], dtype=np.int64)
    np.testing.assert_array_equal(ages_ab, expected, strict=True)

    # And the sum of adjacent age ranges equals the combined range (queried directly)
    np.testing.assert_array_equal(ages_ab + ages_cd, ages_ad, strict=True)


def test_population_by_age_invalid_age_group():
    with pytest.raises(ADRIOProcessingError, match="bad start"):
        # TODO: this isn't a very friendly error message...
        # NOTE: we do need a context to check this, since the age groups
        # reported by the Census may change from year to year.
        acs5.PopulationByAge(19, 20).with_context(
            scope=CountyScope.in_states(["AZ"], year=2020),
            params={"population_by_age_table": acs5.PopulationByAgeTable()},
        ).evaluate()


def test_population_by_age_missing_table():
    with pytest.raises(DataAttributeError, match="missing values"):
        # TODO: should this be a context error?
        # might interfere with missing param reporting in a simulation setting...
        acs5.PopulationByAge(18, 21).with_context(
            scope=CountyScope.in_states(["AZ"], year=2020),
            params={},
        ).evaluate()


@pytest.mark.vcr
def test_population_by_race():
    actual = (
        acs5.PopulationByRace("Black")
        .with_context(scope=TractScope.in_counties(["Santa Cruz, AZ"], year=2015))
        .evaluate()
    )
    expected = np.array([0, 0, 20, 82, 0, 0, 0, 17, 16, 30], dtype=np.int64)
    np.testing.assert_array_equal(actual, expected, strict=True)


########################
# AverageHouseholdSize #
########################


@pytest.mark.vcr
def test_average_household_size():
    actual = (
        acs5.AverageHouseholdSize()
        .with_context(scope=TractScope.in_counties(["Coconino, AZ"], year=2022))
        .evaluate()
    )

    # This data has some sentinel values
    assert np.ma.is_masked(actual)

    # fmt: off
    expected_mask = np.array([
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        True, True, True
    ], dtype=np.bool_)

    expected_data = np.array([
        2.540, 2.220, 2.720, 2.580, 2.700, 2.360, 2.630, 1.880, 2.710, 2.310, 2.740,
        2.200, 1.800, 2.270, 2.490, 2.240, 2.030, 2.700, 2.970, 1.990, 2.030, 1.630,
        2.510, 2.160, 2.080, 3.370, 2.570, 2.420, 2.250, 2.380, 3.370, 3.250, 3.990,
        3.230, 3.310, 3.380, 0.000, 0.000, 0.000,
    ], dtype=np.float64)
    # fmt: on

    expected = np.ma.masked_array(expected_data, expected_mask)
    np.testing.assert_array_equal(actual, expected, strict=True)

    # Try again with fix
    actual = (
        acs5.AverageHouseholdSize(fix_insufficient_data=999.999)
        .with_context(scope=TractScope.in_counties(["Coconino, AZ"], year=2022))
        .evaluate()
    )

    assert not np.ma.is_masked(actual)
    expected = expected_data.copy()
    expected[expected_mask] = 999.999
    np.testing.assert_array_equal(actual, expected, strict=True)


#############
# MedianAge #
#############


@pytest.mark.vcr
def test_median_age():
    actual = (
        acs5.MedianAge()
        .with_context(scope=TractScope.in_counties(["Pinal, AZ"], year=2017))
        .evaluate()
    )
    # fmt: off
    expected = np.array([
        58.800, 33.500, 28.000, 37.700, 31.500, 34.800, 32.700, 29.700, 34.200, 34.100,
        39.800, 32.000, 41.800, 51.100, 46.300, 60.500, 55.000, 55.700, 39.900, 56.000,
        46.900, 46.600, 65.600, 71.300, 57.400, 70.300, 45.400, 67.500, 72.800, 46.800,
        45.300, 51.500, 37.500, 33.000, 43.600, 32.900, 34.500, 45.900, 31.700, 48.000,
        32.400, 39.800, 35.000, 32.300, 36.500, 57.700, 33.000, 40.500, 66.800, 32.100,
        35.700, 42.200, 33.200, 35.500, 35.800, 35.000, 35.500, 33.000, 41.900, 35.800,
        33.500, 32.200, 38.500, 43.700, 34.700, 29.700, 38.700, 46.000, 30.200, 49.800,
        40.700, 47.700, 31.300, 25.700, 34.900,
    ], dtype=np.float64)
    # fmt: on
    np.testing.assert_array_equal(actual, expected, strict=True)


################
# MedianIncome #
################


@pytest.mark.vcr
def test_median_income():
    actual = (
        acs5.MedianIncome()
        .with_context(scope=CountyScope.in_states(["OR"], year=2017))
        .evaluate()
    )
    # fmt: off
    expected = np.array([
        43765, 54682, 72408, 49828, 57449, 40848, 41777, 42519, 59152, 44023,
        39831, 44826, 39504, 57269, 48688, 48464, 40705, 42531, 32769, 47710,
        43291, 49515, 37112, 53828, 54386, 60369, 56032, 42074, 45061, 50071,
        46228, 44877, 48510, 74033, 33563, 58392,
    ], dtype=np.int64)
    # fmt: on
    np.testing.assert_array_equal(actual, expected, strict=True)


#############
# GiniIndex #
#############


@pytest.mark.vcr
def test_gini_index():
    actual = (
        acs5.GiniIndex()
        .with_context(scope=TractScope.in_counties(["Greenlee, AZ"], year=2016))
        .evaluate()
    )
    expected = np.array([0.4310, 0.2782, 0.5094], dtype=np.float64)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_gini_index_invalid_scope():
    with pytest.raises(ADRIOContextError, match="not available for block group scope"):
        (
            acs5.GiniIndex()
            .with_context(
                scope=BlockGroupScope.in_counties(["Coconino, AZ"], year=2016),
            )
            .evaluate()
        )


######################
# DissimilarityIndex #
######################


@pytest.mark.vcr
def test_dissimilarity_index():
    actual = (
        acs5.DissimilarityIndex("White", "Black")
        .with_context(scope=TractScope.in_counties(["Coconino, AZ"], year=2016))
        .evaluate()
    )

    # This data has some sentinel values
    assert np.ma.is_masked(actual)

    # fmt: off
    expected_mask = np.array([
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, True, False, True, True, False, False, False,
        False, False, True, False, True, False, True, True,
    ], dtype=np.bool_)

    # expected is rounded to 6 decimal places
    expected_data = np.array([
        0.707713, 0.253472, 0.376485, 0.578818, 0.629077, 0.585634, 0.293338, 0.249058,
        0.214618, 0.142117, 0.128699, 0.549640, 0.532125, 0.000000, 0.783446, 0.000000,
        0.000000, 0.592673, 0.605825, 0.878195, 0.841504, 0.498290, 0.000000, 0.650000,
        0.000000, 1.000000, 0.000000, 0.000000,
    ], dtype=np.float64)
    # fmt: on

    expected = np.ma.masked_array(expected_data, expected_mask)
    np.testing.assert_array_equal(actual.round(6), expected, strict=True)

    # Try again with fix
    actual = (
        acs5.DissimilarityIndex(
            "White",
            "Black",
            fix_insufficient_population=0,
            fix_not_computable=0.5,
        )
        .with_context(scope=TractScope.in_counties(["Coconino, AZ"], year=2016))
        .evaluate()
    )

    assert not np.ma.is_masked(actual)
    expected = expected_data.copy()
    expected[expected_mask] = 0.5
    np.testing.assert_array_equal(actual.round(6), expected, strict=True)


def test_dissimilarity_index_invalid_scope():
    with pytest.raises(ADRIOContextError, match="not available for block group scope"):
        (
            acs5.DissimilarityIndex("White", "Black")
            .with_context(
                scope=BlockGroupScope.in_counties(["Coconino, AZ"], year=2016),
            )
            .evaluate()
        )
