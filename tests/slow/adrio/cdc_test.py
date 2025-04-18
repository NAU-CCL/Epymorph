import numpy as np
import pytest

from epymorph.adrio import cdc
from epymorph.adrio.processing import RandomFix
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.time import TimeFrame

# NOTE: these tests use VCR to record HTTP requests.
# To re-record this test, load a census API key into the environment, then:
# uv run pytest tests/slow/adrio/cdc_test.py --vcr-mode=record


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["x-app-token"]}


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


@pytest.mark.vcr
def test_influenza_facility_hospitalization_large_request():
    # Let's query many counties: all except 2.
    all_counties = CountyScope.all(year=2019)
    scope = CountyScope.in_counties(list(all_counties.node_ids[0:-2]), year=2019)

    # This should not error-out.
    actual = (
        cdc.InfluenzaFacilityHospitalization(
            fix_missing=0,
            fix_redacted=RandomFix.from_range(1, 3),
        )
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2021-04-04", "2021-04-05"),
            rng=np.random.default_rng(42),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


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


@pytest.mark.vcr
def test_influenza_state_hospitalization_large_request():
    # Let's query many states: all except 1.
    all_states = StateScope.all(year=2019)
    scope = StateScope.in_states(list(all_states.node_ids[0:-1]), year=2019)

    # This should not error-out.
    actual = (
        cdc.InfluenzaStateHospitalization(fix_missing=0)
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2024-04-13", "2024-04-14"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


##########################
# DATA.CDC.GOV 8xkx-amqh #
##########################


@pytest.mark.vcr
def test_covid_vaccination():
    scope = CountyScope.in_counties(
        ["Maricopa, AZ", "Bernalillo, NM", "Hawaii, HI"], year=2019
    )
    time_frame = TimeFrame.rangex("2022-06-15", "2022-07-15")
    actual1 = (
        cdc.COVIDVaccination(
            vaccine_status="at least one dose",
        )
        .with_context(
            scope=scope,
            time_frame=time_frame,
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2022-06-15', 3043892), ('2022-06-15', 0), ('2022-06-15', 582918)],
        [('2022-06-16', 3047711), ('2022-06-16', 0), ('2022-06-16', 582946)],
        [('2022-06-22', 3050657), ('2022-06-22', 0), ('2022-06-22', 583855)],
        [('2022-06-29', 3056883), ('2022-06-29', 0), ('2022-06-29', 585470)],
        [('2022-07-06', 3060841), ('2022-07-06', 0), ('2022-07-06', 586854)],
        [('2022-07-13', 3065313), ('2022-07-13', 0), ('2022-07-13', 587807)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.COVIDVaccination(
            vaccine_status="at least one dose",
            fix_missing=0,
        )
        .with_context(
            scope=scope,
            time_frame=time_frame,
        )
        .evaluate()
    )

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual2), expected1, strict=True)


@pytest.mark.vcr
def test_covid_vaccination_booster():
    scope = CountyScope.in_counties(
        ["Maricopa, AZ", "Bernalillo, NM", "Hawaii, HI"], year=2019
    )
    time_frame = TimeFrame.rangex("2022-06-15", "2022-07-15")
    actual1 = (
        cdc.COVIDVaccination(
            vaccine_status="full series and booster",
        )
        .with_context(
            scope=scope,
            time_frame=time_frame,
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2022-06-15', 1100045), ('2022-06-15', 0), ('2022-06-15', 261725)],
        [('2022-06-16', 1105164), ('2022-06-16', 0), ('2022-06-16', 261744)],
        [('2022-06-22', 1109813), ('2022-06-22', 0), ('2022-06-22', 263073)],
        [('2022-06-29', 1118087), ('2022-06-29', 0), ('2022-06-29', 265846)],
        [('2022-07-06', 1122509), ('2022-07-06', 0), ('2022-07-06', 266788)],
        [('2022-07-13', 1127789), ('2022-07-13', 0), ('2022-07-13', 267630)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
        [(False, False), ( True,  True), (False, False)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)


@pytest.mark.vcr
def test_covid_vaccination_large_request_states():
    # Let's query many states: all except 1.
    all_states = StateScope.all(year=2019)
    scope = StateScope.in_states(list(all_states.node_ids[0:-1]), year=2019)

    # This should not error-out.
    actual = (
        cdc.COVIDVaccination(
            vaccine_status="at least one dose",
            fix_missing=0,
        )
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2022-06-22", "2022-06-23"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


@pytest.mark.vcr
def test_covid_vaccination_large_request_counties():
    # Let's query many counties: all except 2.
    all_counties = CountyScope.all(year=2019)
    scope = CountyScope.in_counties(list(all_counties.node_ids[0:-2]), year=2019)

    # This should not error-out.
    actual = (
        cdc.COVIDVaccination(
            vaccine_status="at least one dose",
            fix_missing=0,
        )
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2022-06-22", "2022-06-23"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


##########################
# DATA.CDC.GOV ite7-j2w7 #
##########################


@pytest.mark.vcr
def test_county_deaths():
    scope = CountyScope.in_states(["AZ"], year=2019)
    time_frame = TimeFrame.rangex("2021-01-01", "2021-02-01")

    actual1 = (
        cdc.CountyDeaths(cause_of_death="COVID-19")
        .with_context(
            scope=scope,
            time_frame=time_frame,
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2021-01-02',  0), ('2021-01-02',  14), ('2021-01-02', 10), ('2021-01-02',  0), ('2021-01-02',   0), ('2021-01-02',  0), ('2021-01-02',  0), ('2021-01-02', 540), ('2021-01-02', 26), ('2021-01-02',  0), ('2021-01-02', 177), ('2021-01-02', 20), ('2021-01-02',  0), ('2021-01-02',  40), ('2021-01-02', 33)],  # noqa: E501
        [('2021-01-09',  0), ('2021-01-09',  13), ('2021-01-09', 12), ('2021-01-09',  0), ('2021-01-09',   0), ('2021-01-09',  0), ('2021-01-09',  0), ('2021-01-09', 615), ('2021-01-09', 35), ('2021-01-09',  0), ('2021-01-09', 164), ('2021-01-09', 23), ('2021-01-09',  0), ('2021-01-09',  31), ('2021-01-09', 28)],  # noqa: E501
        [('2021-01-16',  0), ('2021-01-16',   0), ('2021-01-16',  0), ('2021-01-16',  0), ('2021-01-16',   0), ('2021-01-16',  0), ('2021-01-16',  0), ('2021-01-16', 629), ('2021-01-16', 46), ('2021-01-16',  0), ('2021-01-16', 192), ('2021-01-16', 28), ('2021-01-16',  0), ('2021-01-16',  33), ('2021-01-16', 32)],  # noqa: E501
        [('2021-01-23',  0), ('2021-01-23',   0), ('2021-01-23',  0), ('2021-01-23', 11), ('2021-01-23',   0), ('2021-01-23',  0), ('2021-01-23',  0), ('2021-01-23', 604), ('2021-01-23', 41), ('2021-01-23',  0), ('2021-01-23', 193), ('2021-01-23', 24), ('2021-01-23',  0), ('2021-01-23',  20), ('2021-01-23', 33)],  # noqa: E501
        [('2021-01-30',  0), ('2021-01-30',   0), ('2021-01-30', 12), ('2021-01-30',  0), ('2021-01-30',   0), ('2021-01-30',  0), ('2021-01-30',  0), ('2021-01-30', 514), ('2021-01-30', 39), ('2021-01-30',  0), ('2021-01-30', 137), ('2021-01-30', 19), ('2021-01-30',  0), ('2021-01-30',  29), ('2021-01-30', 15)],  # noqa: E501
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [( True,  True), (False, False), (False, False), ( True,  True), ( True,  True), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False)],  # noqa: E501
        [( True,  True), (False, False), (False, False), ( True,  True), ( True,  True), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False)],  # noqa: E501
        [( True,  True), ( True,  True), ( True,  True), ( True,  True), ( True,  True), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False)],  # noqa: E501
        [( True,  True), ( True,  True), ( True,  True), (False, False), ( True,  True), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False)],  # noqa: E501
        [( True,  True), ( True,  True), (False, False), (False, False), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False), ( True,  True), (False, False), (False, False)],  # noqa: E501
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.CountyDeaths(
            cause_of_death="all",
            fix_redacted=RandomFix.from_range(1, 9),
        )
        .with_context(
            scope=scope,
            time_frame=time_frame,
            rng=np.random.default_rng(42),
        )
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [('2021-01-02',   16), ('2021-01-02',   44), ('2021-01-02',   27), ('2021-01-02',   20), ('2021-01-02',   14), ('2021-01-02',    0), ('2021-01-02',    1), ('2021-01-02', 1367), ('2021-01-02',   94), ('2021-01-02',   22), ('2021-01-02',  416), ('2021-01-02',   75), ('2021-01-02',    7), ('2021-01-02',  117), ('2021-01-02',   76)],  # noqa: E501
        [('2021-01-09',   14), ('2021-01-09',   42), ('2021-01-09',   30), ('2021-01-09',   24), ('2021-01-09',   12), ('2021-01-09',    0), ('2021-01-09',    6), ('2021-01-09', 1468), ('2021-01-09',  113), ('2021-01-09',   30), ('2021-01-09',  432), ('2021-01-09',   89), ('2021-01-09',    4), ('2021-01-09',   96), ('2021-01-09',   63)],  # noqa: E501
        [('2021-01-16',   21), ('2021-01-16',   25), ('2021-01-16',   25), ('2021-01-16',   14), ('2021-01-16',    4), ('2021-01-16',    8), ('2021-01-16',    1), ('2021-01-16', 1475), ('2021-01-16',  129), ('2021-01-16',   27), ('2021-01-16',  469), ('2021-01-16',   96), ('2021-01-16',    7), ('2021-01-16',  100), ('2021-01-16',   62)],  # noqa: E501
        [('2021-01-23',   14), ('2021-01-23',   28), ('2021-01-23',   28), ('2021-01-23',   29), ('2021-01-23',   13), ('2021-01-23',    2), ('2021-01-23',   13), ('2021-01-23', 1432), ('2021-01-23',  114), ('2021-01-23',   36), ('2021-01-23',  427), ('2021-01-23',   95), ('2021-01-23',    1), ('2021-01-23',   93), ('2021-01-23',   73)],  # noqa: E501
        [('2021-01-30',   16), ('2021-01-30',   30), ('2021-01-30',   31), ('2021-01-30',   17), ('2021-01-30',    5), ('2021-01-30',    9), ('2021-01-30',   10), ('2021-01-30', 1288), ('2021-01-30',  108), ('2021-01-30',   23), ('2021-01-30',  332), ('2021-01-30',   74), ('2021-01-30',    7), ('2021-01-30',   94), ('2021-01-30',   46)],  # noqa: E501
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2)
    np.testing.assert_array_equal(actual2, expected2, strict=True)

    # Make sure aggregating to state matches county data.
    actual3 = (
        cdc.CountyDeaths(
            cause_of_death="COVID-19",
            fix_redacted=0,
        )
        .with_context(
            scope=StateScope.in_states(["AZ"], year=2019),
            time_frame=time_frame,
        )
        .evaluate()
    )

    np.testing.assert_array_equal(
        actual3["value"][:, 0],
        np.ma.getdata(actual1)["value"].sum(axis=1),
    )


@pytest.mark.vcr
def test_county_deaths_large_request_states():
    # Let's query many states: all except 1.
    all_states = StateScope.all(year=2019)
    scope = StateScope.in_states(list(all_states.node_ids[0:-1]), year=2019)

    # This should not error-out.
    actual = (
        cdc.CountyDeaths(cause_of_death="COVID-19")
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2021-01-16", "2021-01-17"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


@pytest.mark.vcr
def test_county_deaths_large_request_counties():
    # Let's query many counties: all except 2.
    all_counties = CountyScope.all(year=2019)
    scope = CountyScope.in_counties(list(all_counties.node_ids[0:-2]), year=2019)

    # This should not error-out.
    actual = (
        cdc.CountyDeaths(cause_of_death="COVID-19")
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2021-01-16", "2021-01-17"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)


##########################
# DATA.CDC.GOV r8kw-7aab #
##########################


@pytest.mark.vcr
def test_state_deaths():
    scope = StateScope.in_states(["AZ", "NM", "LA", "ME"], year=2019)
    time_frame = TimeFrame.rangex("2025-02-01", "2025-03-30")

    actual1 = (
        cdc.StateDeaths(cause_of_death="influenza")
        .with_context(
            scope=scope,
            time_frame=time_frame,
        )
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [('2025-02-01', 24), ('2025-02-01',  0), ('2025-02-01',  0), ('2025-02-01', 0)],
        [('2025-02-08', 25), ('2025-02-08',  0), ('2025-02-08',  0), ('2025-02-08', 0)],
        [('2025-02-15', 15), ('2025-02-15',  0), ('2025-02-15', 12), ('2025-02-15', 0)],
        [('2025-02-22',  0), ('2025-02-22',  0), ('2025-02-22',  0), ('2025-02-22', 0)],
        [('2025-03-01', 10), ('2025-03-01', 10), ('2025-03-01', 10), ('2025-03-01', 0)],
        [('2025-03-08', 12), ('2025-03-08',  0), ('2025-03-08', 13), ('2025-03-08', 0)],
        [('2025-03-15',  0), ('2025-03-15',  0), ('2025-03-15',  0), ('2025-03-15', 0)],
        [('2025-03-22',  0), ('2025-03-22',  0), ('2025-03-22', 12), ('2025-03-22', 0)],
        [('2025-03-29',  0), ('2025-03-29',  0), ('2025-03-29',  0), ('2025-03-29', 0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(False, False), ( True,  True), ( True,  True), ( True,  True)],
        [(False, False), ( True,  True), ( True,  True), ( True,  True)],
        [(False, False), ( True,  True), (False, False), ( True,  True)],
        [( True,  True), ( True,  True), ( True,  True), ( True,  True)],
        [(False, False), (False, False), (False, False), ( True,  True)],
        [(False, False), ( True,  True), (False, False), ( True,  True)],
        [( True,  True), ( True,  True), ( True,  True), ( True,  True)],
        [( True,  True), ( True,  True), (False, False), ( True,  True)],
        [( True,  True), ( True,  True), ( True,  True), ( True,  True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.StateDeaths(
            cause_of_death="COVID-19",
            fix_redacted=RandomFix.from_range(1, 9),
        )
        .with_context(
            scope=scope,
            time_frame=time_frame,
            rng=np.random.default_rng(42),
        )
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [('2025-02-01', 13), ('2025-02-01',  1), ('2025-02-01',  7), ('2025-02-01', 6)],
        [('2025-02-08', 12), ('2025-02-08',  4), ('2025-02-08',  4), ('2025-02-08', 8)],
        [('2025-02-15', 12), ('2025-02-15',  1), ('2025-02-15',  7), ('2025-02-15', 2)],
        [('2025-02-22', 21), ('2025-02-22',  1), ('2025-02-22',  5), ('2025-02-22', 9)],
        [('2025-03-01', 12), ('2025-03-01',  7), ('2025-03-01',  7), ('2025-03-01', 7)],
        [('2025-03-08', 11), ('2025-03-08',  8), ('2025-03-08',  5), ('2025-03-08', 2)],
        [('2025-03-15',  8), ('2025-03-15',  5), ('2025-03-15',  5), ('2025-03-15', 4)],
        [('2025-03-22',  2), ('2025-03-22',  9), ('2025-03-22',  8), ('2025-03-22', 6)],
        [('2025-03-29',  4), ('2025-03-29',  8), ('2025-03-29',  0), ('2025-03-29', 5)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2)
    np.testing.assert_array_equal(actual2, expected2, strict=True)


@pytest.mark.vcr
def test_state_deaths_large_request():
    # Let's query many states: all except 1.
    all_states = StateScope.all(year=2019)
    scope = StateScope.in_states(list(all_states.node_ids[0:-1]), year=2019)

    # This should not error-out.
    actual = (
        cdc.StateDeaths(cause_of_death="COVID-19")
        .with_context(
            scope=scope,
            time_frame=TimeFrame.rangex("2025-02-08", "2025-02-09"),
        )
        .evaluate()
    )

    # And we should get an appropriate-shaped response.
    assert actual.shape == (1, scope.nodes)
