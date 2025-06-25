"""ADRIOs that load data from locally available CSV files."""
# ruff: noqa: A005

from itertools import product
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series, read_csv, to_datetime
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOContextError,
    ADRIOProcessingError,
    InspectResult,
    adrio_validate_pipe,
)
from epymorph.adrio.validation import (
    ResultFormat,
    validate_dtype,
    validate_numpy,
    validate_shape_unchecked_arbitrary,
)
from epymorph.data_shape import Shapes
from epymorph.error import MissingContextError
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.geography.us_tiger import get_counties, get_states
from epymorph.simulation import Context, validate_context_for_shape
from epymorph.time import DateRange
from epymorph.util import identity

GeoKeyType = Literal["state_abbrev", "county_state", "geoid"]
"""
The format to use when interpreting values in a geo node identity column.

- `state_abbrev` handles postal codes; e.g., "NY" for New York
- `county_state` handles county name comma state postal code; e.g., "Albany, NY"
- `geoid` handles FIPS code, a.k.a. GEOID, format; e.g., "36001" for Albany, New York
"""


class _CSVMixin(ADRIO):
    key_type: GeoKeyType

    @override
    def validate_context(self, context: Context) -> None:
        try:
            validate_context_for_shape(context, self.result_format.shape)
        except MissingContextError as e:
            raise ADRIOContextError(self, self.context, str(e))

    def parse_geo_key(self, csv_df: DataFrame, key_cols: list[str]) -> DataFrame:
        """
        Convert all columns in `key_cols` using this ADRIO's key type. Returns a copy
        of the `DataFrame` where the values have been converted to GEOID format.
        """
        match self.key_type:
            case "state_abbrev":
                map_keys = self.parse_state_abbrev
            case "county_state":
                map_keys = self.parse_county_state
            case "geoid":
                map_keys = identity

        result_df = csv_df.copy()
        for j in key_cols:
            result_df[j] = map_keys(csv_df[j])
        return result_df

    def parse_state_abbrev(self, keys: Series) -> Series:
        # Parse state postal codes, e.g., AZ
        if not isinstance(self.scope, StateScope):
            err = "State scope is required to use state abbreviation key format."
            raise ADRIOProcessingError(self, self.context, err)

        state_mapping = get_states(self.scope.year).state_code_to_fips
        new_keys = Series([state_mapping.get(x) for x in keys])
        if new_keys.isna().any():
            err = "Invalid state code in key column."
            raise ADRIOProcessingError(self, self.context, err)
        return new_keys

    def parse_county_state(self, keys: Series) -> Series:
        # Parse county name and state postal code, e.g., Maricopa, AZ
        if not isinstance(self.scope, CountyScope):
            err = "County scope is required to use county, state key format."
            raise ADRIOProcessingError(self, self.context, err)

        geoid_to_name = get_counties(self.scope.year).county_fips_to_name
        name_to_geoid = {v: k for k, v in geoid_to_name.items()}
        new_keys = Series([name_to_geoid.get(x) for x in keys])
        if new_keys.isna().any():
            err = "Invalid county name in key column."
            raise ADRIOProcessingError(self, self.context, err)
        return new_keys


class CSVFileN(_CSVMixin, ADRIO[np.generic, np.generic]):
    """
    Loads an N-shaped array of data from a user-provided CSV file.

    Parameters
    ----------
    file_path :
        The path to the CSV file containing data.
    dtype :
        The data type of values in the data column.
    key_col :
        Numerical index of the column containing information to identify geographies.
    key_type :
        The type of geographic identifier in the key column.
    data_col :
        Numerical index of the column containing the data of interest.
    skiprows :
        Number of header rows in the file to be skipped.
    """

    file_path: Path
    """The path to the CSV file containing data."""
    dtype: np.dtype
    """The data type of values in the data column."""
    key_col: int
    """Numerical index of the column containing information to identify geographies."""
    key_type: GeoKeyType
    """The type of geographic identifier in the key column."""
    data_col: int
    """Numerical index of the column containing the data of interest."""
    skiprows: int | None
    """Number of header rows in the file to be skipped."""

    def __init__(
        self,
        *,
        file_path: str | Path,
        dtype: np.dtype | type[np.generic],
        key_col: int,
        key_type: GeoKeyType,
        data_col: int,
        skiprows: int | None = None,
    ):
        if key_col == data_col:
            msg = "Key column and data column must not be the same."
            raise ValueError(msg)

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        self.key_col = key_col
        self.key_type = key_type
        self.data_col = data_col
        self.skiprows = skiprows

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.N, dtype=self.dtype)

    @override
    def inspect(self) -> InspectResult[np.generic, np.generic]:
        self.validate_context(self.context)

        if not self.file_path.is_file():
            err = f"File {self.file_path} not found or not a file."
            raise ADRIOProcessingError(self, self.context, err)

        kwarg_options = {}
        if self.skiprows is not None:
            kwarg_options["skiprows"] = self.skiprows
        csv_df = read_csv(
            self.file_path,
            header=None,
            dtype={self.key_col: str},
            usecols=[self.key_col, self.data_col],
            names=["key", "data"],
            **kwarg_options,
        )

        work_df = self.parse_geo_key(csv_df, ["key"])
        work_df = work_df.sort_values(by="key")
        # Filter to requested geo
        work_df = work_df[work_df["key"].isin(self.scope.node_ids)]

        if not np.array_equal(self.scope.node_ids, work_df["key"]):
            err = (
                "Either required geographies are missing from the CSV file "
                "or some geographies have multiple values."
            )
            raise ADRIOProcessingError(self, self.context, err)
        if work_df["data"].isna().any():
            err = "Data for required geographies missing from CSV file."
            raise ADRIOProcessingError(self, self.context, err)

        result_np = work_df["data"].to_numpy(dtype=self.dtype)
        self.validate_result(self.context, result_np)
        return InspectResult(
            adrio=self,
            source=csv_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )


class CSVFileTxN(_CSVMixin, ADRIO[np.generic, np.generic]):
    """
    Loads a TxN-shaped array of data from a user-provided CSV file.

    Parameters
    ----------
    file_path :
        The path to the CSV file containing data.
    dtype :
        The data type of values in the data column.
    key_col :
        Numerical index of the column containing information to identify geographies.
    key_type :
        The type of geographic identifier in the key column.
    time_col :
        The numerical index of the column containing time information.
    data_col :
        Numerical index of the column containing the data of interest.
    skiprows :
        Number of header rows in the file to be skipped.
    date_range :
        The time period encompassed by data in the file.
    """

    file_path: Path
    """The path to the CSV file containing data."""
    dtype: np.dtype
    """The data type of values in the data column."""
    key_col: int
    """Numerical index of the column containing information to identify geographies."""
    key_type: GeoKeyType
    """The type of geographic identifier in the key column."""
    time_col: int
    """The numerical index of the column containing time information."""
    data_col: int
    """Numerical index of the column containing the data of interest."""
    skiprows: int | None
    """Number of header rows in the file to be skipped."""
    date_range: DateRange | None
    """The time period encompassed by data in the file."""

    def __init__(
        self,
        *,
        file_path: str | Path,
        dtype: np.dtype | type[np.generic],
        key_col: int,
        key_type: GeoKeyType,
        time_col: int,
        data_col: int,
        skiprows: int | None = None,
        date_range: DateRange | None = None,
    ):
        if len({key_col, data_col, time_col}) != 3:
            err = "Key, data, and time columns must all be unique."
            raise ValueError(err)

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        self.key_col = key_col
        self.data_col = data_col
        self.key_type = key_type
        self.skiprows = skiprows
        self.date_range = date_range
        self.time_col = time_col

    @property
    @override
    def result_format(self) -> ResultFormat:
        # result shape is really AxN to support loading irregular time series
        return ResultFormat(shape=Shapes.AxN, dtype=self.dtype)

    @override
    def inspect(self) -> InspectResult:
        self.validate_context(self.context)

        if not self.file_path.is_file():
            err = f"File {self.file_path} not found or not a file."
            raise ADRIOProcessingError(self, self.context, err)

        columns = [self.key_col, self.time_col, self.data_col]
        kwarg_options = {}
        if self.skiprows is not None:
            kwarg_options["skiprows"] = self.skiprows
        csv_df = read_csv(
            self.file_path,
            header=None,
            dtype={self.key_col: str},
            parse_dates=[self.time_col],
            usecols=columns,
            **kwarg_options,
        )

        work_df = csv_df[columns]
        work_df.columns = ["key", "time", "data"]
        work_df = self.parse_geo_key(work_df, ["key"])
        work_df = work_df.sort_values(by=["time", "key"])
        # Filter to requested geo
        work_df = work_df[work_df["key"].isin(self.scope.node_ids)]
        # Filter to specified date range
        # TODO: this should probably just use context time_frame...
        if self.date_range is not None:
            start_date = to_datetime(self.date_range.start_date)
            end_date = to_datetime(self.date_range.end_date)
            work_df = work_df[
                (work_df["time"] >= start_date) & (work_df["time"] <= end_date)
            ]

        nodes_in_data = np.sort(work_df["key"].unique())
        if (
            work_df.duplicated(subset=["key", "time"]).any()  # check duplicate keys
            or not np.array_equal(nodes_in_data, self.scope.node_ids)  # check nodes
        ):
            err = (
                "Either required geographies are missing from the CSV file "
                "or there are some duplicate key/values."
            )
            raise ADRIOProcessingError(self, self.context, err)
        if work_df["data"].isna().any():
            err = "Data for required geographies missing from CSV file."
            raise ADRIOProcessingError(self, self.context, err)

        result_np = work_df.pivot_table(
            index="time",
            columns="key",
            values="data",
            fill_value=None,
        )
        if np.any(np.isnan(result_np)):
            err = (
                "Some data are missing from the CSV file; "
                "not all pairs of date/location are present."
            )
            raise ADRIOProcessingError(self, self.context, err)

        result_np = result_np.to_numpy(dtype=self.dtype)
        self.validate_result(self.context, result_np)
        return InspectResult(
            adrio=self,
            source=csv_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )

    @override
    def validate_result(self, context: Context, result: NDArray[np.generic]) -> None:
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape_unchecked_arbitrary(
                self.result_format.shape.to_tuple(context.dim)
            ),
            validate_dtype(self.dtype),
        )


class CSVFileNxN(_CSVMixin, ADRIO[np.generic, np.generic]):
    """
    Loads an NxN-shaped array of data from a user-provided CSV file.

    Parameters
    ----------
    file_path :
        The path to the CSV file containing data.
    dtype :
        The data type of values in the data column.
    from_key_col :
        Index of the column identifying source geographies.
    to_key_col :
        Index of the column identifying destination geographies.
    key_type :
        The type of geographic identifier in the key columns.
    data_col :
        Index of the column containing the data of interest.
    skiprows :
        Number of header rows in the file to be skipped.
    """

    file_path: Path
    """The path to the CSV file containing data."""
    dtype: np.dtype
    """The data type of values in the data column."""
    from_key_col: int
    """Index of the column identifying source geographies."""
    to_key_col: int
    """Index of the column identifying destination geographies."""
    key_type: GeoKeyType
    """The type of geographic identifier in the key columns."""
    data_col: int
    """Index of the column containing the data of interest."""
    skiprows: int | None
    """Number of header rows in the file to be skipped."""

    def __init__(
        self,
        *,
        file_path: PathLike,
        dtype: np.dtype | type[np.generic],
        from_key_col: int,
        to_key_col: int,
        key_type: GeoKeyType,
        data_col: int,
        skiprows: int | None,
    ):
        if len({from_key_col, to_key_col, data_col}) != 3:
            err = "From key column, to key column, and data column must all be unique."
            raise ValueError(err)

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        self.from_key_col = from_key_col
        self.to_key_col = to_key_col
        self.key_type = key_type
        self.data_col = data_col
        self.skiprows = skiprows

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.NxN, dtype=self.dtype)

    @override
    def inspect(self) -> InspectResult:
        self.validate_context(self.context)

        if not self.file_path.is_file():
            err = f"File {self.file_path} not found or not a file."
            raise ADRIOProcessingError(self, self.context, err)

        kwarg_options = {}
        if self.skiprows is not None:
            kwarg_options["skiprows"] = self.skiprows
        csv_df = read_csv(
            self.file_path,
            header=None,
            dtype={self.from_key_col: str, self.to_key_col: str},
            usecols=[self.from_key_col, self.to_key_col, self.data_col],
            names=["from_key", "to_key", "data"],
            **kwarg_options,
        )

        work_df = self.parse_geo_key(csv_df, ["from_key", "to_key"])
        work_df = work_df.sort_values(by=["from_key", "to_key"])

        # Filter to requested geo
        node_ids = self.scope.node_ids
        work_df = work_df[
            work_df["from_key"].isin(node_ids) & work_df["to_key"].isin(node_ids)
        ]

        n = self.scope.nodes
        if (n * n) != len(work_df):
            err = (
                "Either required geographies are missing from the CSV file "
                "or some geographies have multiple values."
            )
            raise ADRIOProcessingError(self, self.context, err)

        expected_ids = product(node_ids, node_ids)
        rows = work_df.itertuples(index=False, name=None)
        if any(
            a != x or b != y
            for (a, b), (x, y, _) in zip(expected_ids, rows, strict=True)
        ):
            err = (
                "Either required geographies are missing from the CSV file "
                "or some geographies have multiple values."
            )
            raise ADRIOProcessingError(self, self.context, err)

        if work_df["data"].isna().any():
            err = "Data for required geographies missing from CSV file."
            raise ADRIOProcessingError(self, self.context, err)

        result_np = work_df.pivot_table(
            index="from_key",
            columns="to_key",
            values="data",
            sort=True,
        ).to_numpy(dtype=self.dtype)
        self.validate_result(self.context, result_np)
        return InspectResult(
            adrio=self,
            source=csv_df,
            result=result_np,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues={},
        )
