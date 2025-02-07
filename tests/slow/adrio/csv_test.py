from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymorph.adrio import csv
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.geography.us_tiger import get_counties, get_states
from epymorph.time import TimeFrame


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_csv_01(tmp_path, rng):
    """
    Tests a basic csv file using state abbreviation as the geo key.
    No header row. CSV ADRIO loads a geo subset of the available data
    to test filtering.
    """
    tmp_file = tmp_path / "population.csv"

    scope = StateScope.in_states(
        ["AZ", "FL", "GA", "MD", "NY", "NC", "SC", "VA"],
        year=2015,
    )
    to_postal_code = get_states(2015).state_fips_to_code

    # the values are arbitrary, but should align with scope
    population = rng.uniform(1_000, 1_000_000, size=scope.nodes).astype(np.int64)

    # write csv file
    data_df = pd.DataFrame(
        {
            "label": [to_postal_code[x] for x in scope.node_ids],
            "population": population,
        }
    ).sample(frac=1, random_state=rng)  # put the rows in a (seeded) random order
    data_df.to_csv(tmp_file, header=False, index=False)

    # load the data
    actual = (
        csv.CSV(
            file_path=tmp_file,
            key_col=0,
            data_col=1,
            data_type=np.int64,
            key_type="state_abbrev",
            skiprows=None,
        )
        .with_context(
            # NOTE: to test filtering, load back a geographic subset of the initial data
            # this scope is the same as above but minus AZ and NY
            scope=StateScope.in_states(["12", "13", "24", "37", "45", "51"], year=2015),
        )
        .evaluate()
    )

    # compare with expected values
    expected = population[[1, 2, 3, 5, 6, 7]]

    assert np.array_equal(actual, expected)


def test_csv_02(tmp_path, rng):
    """
    Tests a csv file with a header and multiple data columns.
    Uses county/state as the geo key.
    """
    tmp_file = tmp_path / "population.csv"

    scope = CountyScope.in_states(["AZ", "NM"], year=2015)
    to_county_name = get_counties(2015).county_fips_to_name

    population = rng.uniform(0, 100_000, size=(scope.nodes, 3)).astype(np.int64)
    data_df = pd.DataFrame(
        {
            "Date": date(2015, 1, 1),
            "County": [to_county_name[x] for x in scope.node_ids],
            "Young": population[:, 0],
            "Adult": population[:, 1],
            "Elderly": population[:, 2],
        }
    ).sample(frac=1, random_state=rng)
    data_df.to_csv(tmp_file, index=False)

    young = (
        csv.CSV(
            file_path=tmp_file,
            key_col=1,
            data_col=2,
            data_type=np.int64,
            key_type="county_state",
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )
    adult = (
        csv.CSV(
            file_path=tmp_file,
            key_col=1,
            data_col=3,
            data_type=np.int64,
            key_type="county_state",
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )
    elderly = (
        csv.CSV(
            file_path=tmp_file,
            key_col=1,
            data_col=4,
            data_type=np.int64,
            key_type="county_state",
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )

    assert np.array_equal(young, population[:, 0])
    assert np.array_equal(adult, population[:, 1])
    assert np.array_equal(elderly, population[:, 2])


def test_csv_03(tmp_path, rng):
    """
    Tests CSVTimeSeries ADRIO for mock vaccination data.
    Uses GEOID as the geo key.
    """
    tmp_file = tmp_path / "vaccines.csv"

    data_scope = CountyScope.in_counties(
        ["08001", "35001", "04013", "04017"],
        year=2021,
    )
    date_range = pd.date_range(start="2021-01-01", end="2021-03-31", freq="D")
    data_df = pd.DataFrame(
        [(date, fips) for fips in data_scope.node_ids for date in date_range],
        columns=["date", "fips"],
    )
    data_df["series_complete_yes"] = np.floor(rng.uniform(0, 100000, size=len(data_df)))

    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, index=False)

    adrio_scope = CountyScope.in_counties(["08001", "35001", "04013"], year=2021)

    # NOTE: time subsetting is not currently supported by the CSVTimeSeries ADRIO.
    # adrio_time_frame = TimeFrame.range("2021-01-15", "2021-03-15")  # noqa: ERA001
    adrio_time_frame = TimeFrame.range("2021-01-01", "2021-03-31")

    adrio_date_range = pd.date_range(
        adrio_time_frame.start_date,
        adrio_time_frame.end_date,
        freq="D",
    )
    txn = (adrio_time_frame.days, adrio_scope.nodes)

    actual = (
        csv.CSVTimeSeries(
            file_path=tmp_file,
            time_col=0,
            time_frame=adrio_time_frame,
            key_col=1,
            data_col=2,
            data_type=np.float64,
            key_type="geoid",
            skiprows=1,
        )
        .with_context(scope=adrio_scope, time_frame=adrio_time_frame)
        .evaluate()
    )

    expected = (
        data_df[
            data_df["fips"].isin(adrio_scope.node_ids)
            & data_df["date"].isin(adrio_date_range)
        ]
        .pivot_table(index="date", columns="fips", values="series_complete_yes")
        .to_numpy()
    )

    assert actual.shape == txn
    assert np.array_equal(actual, expected)


def test_csv_04(tmp_path, rng):
    """
    Test CSVMatrix with mock commuters data.
    Tests geo subsetting and uses GEOID as the geo key.
    """
    tmp_file = tmp_path / "commuters.csv"

    data_scope = CountyScope.in_counties(
        ["08001", "35001", "04013", "04017"],
        year=2020,
    )
    commuters = rng.uniform(0, 5000, size=(data_scope.nodes, data_scope.nodes)).astype(
        np.int64
    )

    home, work = np.meshgrid(data_scope.node_ids, data_scope.node_ids, indexing="ij")
    data_df = pd.DataFrame(
        {
            "res_geoid": home.flatten(),
            "wrk_geoid": work.flatten(),
            "workers": commuters.flatten(),
        }
    )
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, index=False)

    adrio_scope = CountyScope.in_counties(["35001", "04013", "04017"], year=2020)

    actual = (
        csv.CSVMatrix(
            file_path=tmp_file,
            from_key_col=0,
            to_key_col=1,
            data_col=2,
            data_type=np.int64,
            key_type="geoid",
            skiprows=1,
        )
        .with_context(scope=adrio_scope)
        .evaluate()
    )

    geo_subset = [x in adrio_scope.node_ids for x in data_scope.node_ids]
    expected = commuters[geo_subset, :][:, geo_subset]
    assert np.array_equal(actual, expected)
