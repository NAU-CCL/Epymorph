import numpy as np
import pytest

from epymorph.adrio import cdc
from epymorph.adrio.processing import RandomFix
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.time import TimeFrame

# NOTE: these tests use VCR to record HTTP requests.
# To re-record this test load a census API key into the environment, then:
# uv run pytest tests/slow/adrio/cdc_test.py --record-mode=rewrite


@pytest.fixture(scope="module")
def vcr_config(global_vcr_config):
    return {**global_vcr_config, "filter_headers": ["x-app-token"]}


############################
# HEALTHDATA.GOV anag-cw7u #
############################


@pytest.mark.vcr
def test_covid_facility_hospitalization():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    actual1 = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [("2021-04-04", 0), ("2021-04-04", 0)],
        [("2021-04-11", 0), ("2021-04-11", 0)],
        [("2021-04-18", 0), ("2021-04-18", 430)],
        [("2021-04-25", 0), ("2021-04-25", 399)],
        [("2021-05-02", 0), ("2021-05-02", 449)],
        [("2021-05-09", 1991), ("2021-05-09", 457)],
        [("2021-05-16", 0), ("2021-05-16", 0)],
        [("2021-05-23", 0), ("2021-05-23", 0)],
        [("2021-05-30", 0), ("2021-05-30", 0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(True, True), (True, True)],
        [(True, True), (True, True)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(False, False), (False, False)],
        [(True, True), (True, True)],
        [(True, True), (True, True)],
        [(True, True), (True, True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
            fix_missing=0,
            fix_redacted=RandomFix.from_range(1, 3),
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [("2021-04-04", 1609), ("2021-04-04", 372)],
        [("2021-04-11", 1685), ("2021-04-11", 419)],
        [("2021-04-18", 1808), ("2021-04-18", 430)],
        [("2021-04-25", 2044), ("2021-04-25", 399)],
        [("2021-05-02", 1956), ("2021-05-02", 449)],
        [("2021-05-09", 1991), ("2021-05-09", 457)],
        [("2021-05-16", 2027), ("2021-05-16", 473)],
        [("2021-05-23", 1971), ("2021-05-23", 354)],
        [("2021-05-30", 1650), ("2021-05-30", 327)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(actual2, expected2, strict=True)

    # The unmasked values from the first request
    # should be unchanged in the second request.
    mask = np.ma.getmask(actual2["value"])
    np.testing.assert_array_equal(actual1["value"][mask], actual2["value"][mask])


@pytest.mark.vcr
def test_covid_facility_hospitalization_age_groups():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    # The pediatric and adult cases should add up to the total,
    # but only if we remove random values.
    actual_both = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    actual_adult = (
        cdc.COVIDFacilityHospitalization(
            age_group="adult",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    actual_pediatric = (
        cdc.COVIDFacilityHospitalization(
            age_group="pediatric",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    np.testing.assert_array_equal(
        actual_adult["value"] + actual_pediatric["value"],
        actual_both["value"],
        strict=True,
    )


@pytest.mark.vcr
def test_influenza_facility_hospitalization():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    actual1 = (
        cdc.InfluenzaFacilityHospitalization().with_context(**context()).evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [("2021-04-04", 53), ("2021-04-04", 0)],
        [("2021-04-11", 58), ("2021-04-11", 0)],
        [("2021-04-18", 56), ("2021-04-18", 0)],
        [("2021-04-25", 59), ("2021-04-25", 0)],
        [("2021-05-02", 56), ("2021-05-02", 0)],
        [("2021-05-09", 49), ("2021-05-09", 0)],
        [("2021-05-16", 65), ("2021-05-16", 0)],
        [("2021-05-23", 67), ("2021-05-23", 0)],
        [("2021-05-30", 53), ("2021-05-30", 0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(True, True), (True, True)],
        [(False, False), (True, True)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(False, False), (True, True)],
        [(True, True), (True, True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.InfluenzaFacilityHospitalization(
            fix_missing=0,
            fix_redacted=RandomFix.from_range(1, 3),
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [("2021-04-04", 54), ("2021-04-04", 3)],
        [("2021-04-11", 58), ("2021-04-11", 2)],
        [("2021-04-18", 56), ("2021-04-18", 0)],
        [("2021-04-25", 59), ("2021-04-25", 0)],
        [("2021-05-02", 58), ("2021-05-02", 0)],
        [("2021-05-09", 54), ("2021-05-09", 0)],
        [("2021-05-16", 66), ("2021-05-16", 0)],
        [("2021-05-23", 67), ("2021-05-23", 3)],
        [("2021-05-30", 57), ("2021-05-30", 3)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(actual2, expected2, strict=True)

    # The unmasked values from the first request
    # should be unchanged in the second request.
    mask = np.ma.getmask(actual2["value"])
    np.testing.assert_array_equal(actual1["value"][mask], actual2["value"][mask])


# TODO: test "all" geo

##########################
# DATA.CDC.GOV 3nnm-4jni #
##########################


@pytest.mark.vcr
def test_covid_county_cases():
    actual1 = (
        cdc.COVIDCountyCases()
        .with_context(
            scope=CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            time_frame=TimeFrame.rangex("2022-02-24", "2022-05-11"),
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2022-02-24', 8552), ('2022-02-24', 1288)],
        [('2022-03-03', 3475), ('2022-03-03',  879)],
        [('2022-03-10', 7107), ('2022-03-10',  503)],
        [('2022-03-17', 3368), ('2022-03-17',  419)],
        [('2022-03-24', 3359), ('2022-03-24',  327)],
        [('2022-03-31', 8281), ('2022-03-31',  273)],
        [('2022-04-07', 4706), ('2022-04-07',  265)],
        [('2022-04-14',  652), ('2022-04-14',  259)],
        [('2022-04-21', 1874), ('2022-04-21',  387)],
        [('2022-04-28', 1705), ('2022-04-28',  321)],
        [('2022-05-05', 2642), ('2022-05-05',  696)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(actual1, expected1, strict=True)

    actual2 = (
        cdc.COVIDCountyCases()
        .with_context(
            scope=StateScope.in_states(["AZ", "NM"], year=2019),
            time_frame=TimeFrame.rangex("2022-02-24", "2022-05-11"),
        )
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [('2022-02-24', 14593), ('2022-02-24',  4278)],
        [('2022-03-03',  5212), ('2022-03-03',  2820)],
        [('2022-03-10', 10428), ('2022-03-10',  1870)],
        [('2022-03-17',  5153), ('2022-03-17',  1338)],
        [('2022-03-24',  4566), ('2022-03-24',  1045)],
        [('2022-03-31', 10143), ('2022-03-31',   855)],
        [('2022-04-07',  6840), ('2022-04-07',   955)],
        [('2022-04-14',  2789), ('2022-04-14',   799)],
        [('2022-04-21',  2445), ('2022-04-21',  1074)],
        [('2022-04-28',  2358), ('2022-04-28',   832)],
        [('2022-05-05',  3911), ('2022-05-05',  1831)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(actual2, expected2, strict=True)


@pytest.mark.vcr
def test_covid_county_cases_large_request():
    # NOTE: it's a little weird to VCR this one, since the error was server-side
    # (414 response). But at least this proves it worked once.
    # *Truly* testing this requires re-recording the casette however.

    # Let's query many counties: all except 2.
    all_counties = CountyScope.all(year=2019)
    scope = CountyScope.in_counties(list(all_counties.node_ids[0:-2]), year=2019)

    # This should not error-out.
    actual = (
        cdc.COVIDCountyCases()
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2022-02-24", "2022-02-25"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


##########################
# DATA.CDC.GOV aemt-mg7g #
##########################


@pytest.mark.vcr
def test_covid_state_hospitalization():
    actual1 = (
        cdc.COVIDStateHospitalization()
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2024-04-06', 117), ('2024-04-06', 162)],
        [('2024-04-13', 125), ('2024-04-13', 157)],
        [('2024-04-20',  96), ('2024-04-20', 124)],
        [('2024-04-27',  90), ('2024-04-27', 111)],
        [('2024-05-04', 104), ('2024-05-04', 119)],
        [('2024-05-11',  78), ('2024-05-11',  54)],
        [('2024-05-18',  72), ('2024-05-18',   0)],
        [('2024-05-25',  93), ('2024-05-25',   0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), ( True,  True)],
        [(False, False), ( True,  True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.COVIDStateHospitalization(
            allow_voluntary=False,
        )
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected2 = expected1
    expected_mask2 = np.array([
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual2), expected2, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual2), expected_mask2, strict=True)

    actual3 = (
        cdc.COVIDStateHospitalization(fix_missing=0)
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected3 = expected1
    # fmt: on

    assert not np.ma.is_masked(actual3["value"])
    np.testing.assert_array_equal(actual3, expected3, strict=True)


@pytest.mark.vcr
def test_influenza_state_hospitalization():
    actual1 = (
        cdc.InfluenzaStateHospitalization()
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2024-04-06',  83), ('2024-04-06', 214)],
        [('2024-04-13',  84), ('2024-04-13', 168)],
        [('2024-04-20',  79), ('2024-04-20', 125)],
        [('2024-04-27', 105), ('2024-04-27',  65)],
        [('2024-05-04',  73), ('2024-05-04',  58)],
        [('2024-05-11',  68), ('2024-05-11',  48)],
        [('2024-05-18',  63), ('2024-05-18',   0)],
        [('2024-05-25',  66), ('2024-05-25',   0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), ( True,  True)],
        [(False, False), ( True,  True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.InfluenzaStateHospitalization(
            allow_voluntary=False,
        )
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected2 = expected1
    expected_mask2 = np.array([
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
        [( True,  True), ( True,  True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual2), expected2, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual2), expected_mask2, strict=True)

    actual3 = (
        cdc.InfluenzaStateHospitalization(fix_missing=0)
        .with_context(
            scope=StateScope.in_states(["AZ", "MA"], year=2019),
            time_frame=TimeFrame.rangex("2024-04-01", "2024-06-01"),
        )
        .evaluate()
    )

    # fmt: off
    expected3 = expected1
    # fmt: on

    assert not np.ma.is_masked(actual3["value"])
    np.testing.assert_array_equal(actual3, expected3, strict=True)


# TODO: test "all" geo

##########################
# DATA.CDC.GOV 8xkx-amqh #
##########################

# TODO: def test_covid_vaccination():
# TODO: test "all" geo

##########################
# DATA.CDC.GOV ite7-j2w7 #
##########################

# TODO: def test_county_deaths():
# TODO: test "all" geo

##########################
# DATA.CDC.GOV r8kw-7aab #
##########################

# TODO: def test_state_deaths():
# TODO: test "all" geo
