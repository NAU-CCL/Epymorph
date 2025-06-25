from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymorph.adrio import csv
from epymorph.adrio.adrio import ADRIOProcessingError
from epymorph.geography.custom import CustomScope
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.geography.us_tiger import get_counties, get_states
from epymorph.time import DateRange, TimeFrame


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _generate_n_data(rng, nodes):
    population = rng.integers(0, 100_000, size=(len(nodes), 3))
    return pd.DataFrame(
        {
            "Date": date(2015, 1, 1),  # here to have a column to ignore
            "Node": nodes,
            "Young": population[:, 0],
            "Adult": population[:, 1],
            "Elderly": population[:, 2],
        }
    )


def test_csv_n_01(tmp_path, rng):
    # CSVFileN successful case
    # - Uses state_abbrev as the geo key
    data_scope = StateScope.in_states(
        ["AZ", "FL", "GA", "MD", "NY", "NC", "SC", "VA"],
        year=2015,
    )
    to_postal_code = get_states(2015).state_fips_to_code
    nodes = [to_postal_code[x] for x in data_scope.node_ids]

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, nodes)
    # scramble row order when writing file
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, header=False, index=False)

    # NOTE: to test filtering, load a geographic subset of the original data
    # this scope is the same as above but minus AZ and NY
    subset_nodes = ["FL", "GA", "MD", "NC", "SC", "VA"]
    actual = (
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="state_abbrev",
            data_col=3,
        )
        .with_context(scope=StateScope.in_states(subset_nodes, year=2015))
        .evaluate()
    )

    expected = data_df[data_df["Node"].isin(subset_nodes)]

    assert np.array_equal(actual, expected["Adult"].to_numpy())


def test_csv_n_02(tmp_path, rng):
    # CSVFileN successful case
    # - Data has a header and multiple data columns
    # - Uses county/state as the geo key
    scope = CountyScope.in_states(["AZ", "NM"], year=2015)
    to_county_name = get_counties(2015).county_fips_to_name
    nodes = [to_county_name[x] for x in scope.node_ids]

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, nodes)
    # scramble row order when writing file
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, header=True, index=False)

    young = (
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="county_state",
            data_col=2,
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )
    adult = (
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="county_state",
            data_col=3,
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )
    elderly = (
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="county_state",
            data_col=4,
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )

    expected = data_df

    assert np.array_equal(young, expected["Young"].to_numpy())
    assert np.array_equal(adult, expected["Adult"].to_numpy())
    assert np.array_equal(elderly, expected["Elderly"].to_numpy())


def test_csv_n_03(tmp_path, rng):
    # Error: CSVFileN duplicated data
    scope = StateScope.in_states(["AZ", "NM"], year=2015)

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, scope.node_ids)
    data_df = pd.concat((data_df, data_df.iloc[[0]]))
    data_df.to_csv(tmp_file, header=False, index=False)

    with pytest.raises(ADRIOProcessingError, match="geographies have multiple values"):
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            data_col=3,
        ).with_context(scope=scope).evaluate()


def test_csv_n_04(tmp_path, rng):
    # Error: CSVFileN missing data
    scope = StateScope.in_states(["AZ", "NM"], year=2015)

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, scope.node_ids)
    data_df = data_df.drop(0)
    data_df.to_csv(tmp_file, header=False, index=False)

    with pytest.raises(ADRIOProcessingError, match="geographies are missing"):
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            data_col=3,
        ).with_context(scope=scope).evaluate()


def test_csv_n_05(tmp_path, rng):
    # Error: CSVFileN incorrect key type
    scope = StateScope.in_states(["AZ", "NM"], year=2015)

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, scope.node_ids)
    data_df.to_csv(tmp_file, header=False, index=False)

    with pytest.raises(ADRIOProcessingError, match="Invalid state code"):
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="state_abbrev",
            data_col=3,
        ).with_context(scope=scope).evaluate()


def test_csv_n_06(tmp_path):
    # Error: CSVFileN same columns
    with pytest.raises(ValueError, match="column must not be the same"):
        csv.CSVFileN(
            file_path=tmp_path,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            data_col=1,
        )


def test_csv_n_07(tmp_path):
    # Error: CSVFileN missing file
    scope = StateScope.in_states(["AZ", "NM"], year=2015)
    with pytest.raises(ADRIOProcessingError, match="foo.csv not found"):
        csv.CSVFileN(
            file_path=tmp_path / "foo.csv",
            dtype=np.int64,
            key_col=1,
            key_type="state_abbrev",
            data_col=3,
        ).with_context(scope=scope).evaluate()


def test_csv_n_08(tmp_path, rng):
    # CSVFileN with CustomScope
    scope = CustomScope(["A", "B", "C"])

    tmp_file = tmp_path / "population.csv"
    data_df = _generate_n_data(rng, scope.node_ids)
    # scramble row order when writing file
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, header=False, index=False)

    actual = (
        csv.CSVFileN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            data_col=3,
        )
        .with_context(scope=scope)
        .evaluate()
    )

    expected = data_df
    assert np.array_equal(actual, expected["Adult"].to_numpy())


def _generate_txn_data(rng):
    return pd.DataFrame(
        [
            (date, fips, rng.integers(0, 100000))
            for fips in ["08001", "35001", "04013", "04017"]
            for date in pd.date_range(start="2021-01-01", end="2021-03-31", freq="2D")
        ],
        columns=["date", "fips", "series_complete_yes"],
    )


def test_csv_txn_01(tmp_path, rng):
    # CSVFileTxN successful case
    tmp_file = tmp_path / "vaccines.csv"
    data_df = _generate_txn_data(rng)
    # scramble row order when writing file
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, index=False)

    # time and geo subsetting!
    scope = CountyScope.in_counties(["08001", "35001", "04013"], year=2021)
    time_frame = TimeFrame.range("2021-01-15", "2021-03-14")
    date_range = DateRange(time_frame.start_date, time_frame.end_date, step=2)

    actual = (
        csv.CSVFileTxN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            time_col=0,
            data_col=2,
            skiprows=1,
            date_range=date_range,
        )
        .with_context(scope=scope)
        .evaluate()
    )

    pd_date_range = date_range.to_pandas()
    subset_df = data_df[
        data_df["fips"].isin(scope.node_ids) & data_df["date"].isin(pd_date_range)
    ]
    expected = subset_df.pivot_table(
        index="date",
        columns="fips",
        values="series_complete_yes",
        sort=True,
    ).to_numpy(dtype=np.int64)

    assert actual.shape == (len(pd_date_range), scope.nodes)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_csv_txn_02(tmp_path, rng):
    # Error: data contains duplicates
    tmp_file = tmp_path / "vaccines.csv"
    data_df = _generate_txn_data(rng)
    # find index of a row in the date range and duplicate it
    i = (data_df.index[data_df["date"] == pd.to_datetime("2021-01-15")])[0]
    data_df = pd.concat((data_df, data_df.iloc[[i]]))
    data_df.to_csv(tmp_file, index=False)

    scope = CountyScope.in_counties(["08001", "35001", "04013"], year=2021)
    time_frame = TimeFrame.range("2021-01-15", "2021-03-14")
    date_range = DateRange(time_frame.start_date, time_frame.end_date)

    with pytest.raises(ADRIOProcessingError, match="duplicate key/values"):
        csv.CSVFileTxN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            time_col=0,
            data_col=2,
            skiprows=1,
            date_range=date_range,
        ).with_context(scope=scope).evaluate()


def test_csv_txn_03(tmp_path, rng):
    # Error: data missing
    tmp_file = tmp_path / "vaccines.csv"
    data_df = _generate_txn_data(rng)
    # find index of a row in the date range and drop it
    i = (data_df.index[data_df["date"] == pd.to_datetime("2021-01-15")])[0]
    data_df = data_df.drop(i)
    data_df.to_csv(tmp_file, index=False)

    scope = CountyScope.in_counties(["08001", "35001", "04013", "04017"], year=2021)
    time_frame = TimeFrame.range("2021-01-15", "2021-03-14")
    date_range = DateRange(time_frame.start_date, time_frame.end_date)

    with pytest.raises(ADRIOProcessingError, match="data are missing"):
        csv.CSVFileTxN(
            file_path=tmp_file,
            dtype=np.int64,
            key_col=1,
            key_type="geoid",
            time_col=0,
            data_col=2,
            skiprows=1,
            date_range=date_range,
        ).with_context(scope=scope).evaluate()


def _generate_nxn_data(rng):
    node_ids = ["08001", "35001", "04013", "04017"]
    commuters = rng.integers(0, 5000, size=(len(node_ids), len(node_ids)))
    home, work = np.meshgrid(node_ids, node_ids, indexing="ij")
    return pd.DataFrame(
        {
            "res_geoid": home.flatten(),
            "wrk_geoid": work.flatten(),
            "workers": commuters.flatten(),
        }
    )


def test_csv_nxn_01(tmp_path, rng):
    # CSVFileNxN successful case
    tmp_file = tmp_path / "commuters.csv"
    data_df = _generate_nxn_data(rng)
    # scramble row order when writing file
    data_df.sample(frac=1, random_state=rng).to_csv(tmp_file, index=False)

    scope = CountyScope.in_counties(["08001", "35001", "04017"], year=2020)

    actual = (
        csv.CSVFileNxN(
            file_path=tmp_file,
            dtype=np.int64,
            from_key_col=0,
            to_key_col=1,
            key_type="geoid",
            data_col=2,
            skiprows=1,
        )
        .with_context(scope=scope)
        .evaluate()
    )

    subset_df = data_df[
        data_df["res_geoid"].isin(scope.node_ids)
        & data_df["wrk_geoid"].isin(scope.node_ids)
    ]
    expected = subset_df.pivot_table(
        index="res_geoid",
        columns="wrk_geoid",
        values="workers",
        sort=True,
    ).to_numpy(dtype=np.int64)

    assert actual.shape == (scope.nodes, scope.nodes)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_csv_nxn_02(tmp_path, rng):
    # Error: data contains duplicates
    tmp_file = tmp_path / "commuters.csv"
    data_df = _generate_nxn_data(rng)
    data_df = pd.concat((data_df, data_df.iloc[[0]]))
    data_df.to_csv(tmp_file, index=False)

    scope = CountyScope.in_counties(["08001", "35001", "04017"], year=2020)

    with pytest.raises(ADRIOProcessingError, match="geographies have multiple values"):
        csv.CSVFileNxN(
            file_path=tmp_file,
            dtype=np.int64,
            from_key_col=0,
            to_key_col=1,
            key_type="geoid",
            data_col=2,
            skiprows=1,
        ).with_context(scope=scope).evaluate()


def test_csv_nxn_03(tmp_path, rng):
    # Error: data missing
    tmp_file = tmp_path / "commuters.csv"
    data_df = _generate_nxn_data(rng)
    data_df = data_df[1:]
    data_df.to_csv(tmp_file, index=False)

    scope = CountyScope.in_counties(["08001", "35001", "04017"], year=2020)

    with pytest.raises(ADRIOProcessingError, match="geographies are missing"):
        csv.CSVFileNxN(
            file_path=tmp_file,
            dtype=np.int64,
            from_key_col=0,
            to_key_col=1,
            key_type="geoid",
            data_col=2,
            skiprows=1,
        ).with_context(scope=scope).evaluate()
