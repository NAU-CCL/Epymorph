import dataclasses
import os
from functools import partial
from typing import Callable, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

import epymorph.adrio.soda as q
from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    FetchADRIO,
    ProcessResult,
    ResultFormat,
    range_mask_fn,
    validate_time_frame,
)
from epymorph.adrio.processing import (
    DateValueType,
    Fill,
    Fix,
    FixSentinel,
    process_txn,
)
from epymorph.data_shape import Shapes
from epymorph.geography.us_census import CensusScope, CountyScope, StateScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import get_states
from epymorph.simulation import Context
from epymorph.time import DateRange, iso8601


def healthdata_api_key() -> str | None:
    """
    Loads the Socrata API key to use for healthdata.gov,
    as environment variable 'API_KEY__healthdata.gov'.
    """
    return os.environ.get("API_KEY__healthdata.gov", default=None)


def data_cdc_api_key() -> str | None:
    """
    Loads the Socrata API key to use for data.cdc.gov,
    as environment variable 'API_KEY__data.cdc.gov'.
    """
    return os.environ.get("API_KEY__data.cdc.gov", default=None)


############################
# HEALTHDATA.GOV anag-cw7u #
############################


class _HealthdataAnagCw7u(FetchADRIO[DateValueType, np.int64]):
    """
    An abstract specialization of `FetchADRIO` for ADRIOs which fetch
    data from healthdata.gov dataset anag-cw7u: a.k.a.
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility".
    https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
    """

    _RESOURCE = q.SocrataResource(domain="healthdata.gov", id="anag-cw7u")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), step=7)
    """The time range over which values are available."""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.AxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=True,
    )
    """The format of results."""

    _REDACTED_VALUE = np.int64(-999999)
    """The value of redacted reports: between 1 and 3 cases."""

    _fix_redacted: Fix[np.int64]
    """The method to use to replace redacted values (-999999 in the data)."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        *,
        fix_redacted: Fix[np.int64] | int | Callable[[], int] | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            self._fix_redacted = Fix.of_int64(fix_redacted)
        except ValueError:
            raise ValueError("Invalid value for `fix_redacted`")
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return self._RESULT_FORMAT

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


class COVIDFacilityHospitalization(_HealthdataAnagCw7u):
    """
    Loads COVID hospitalization data from HealthData.gov's
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"
    dataset. The data were reported by healthcare facilities on a weekly basis,
    starting 2019-12-29 and ending 2024-04-21, although the data is not complete
    over this entire range, nor over the entire United States.

    This ADRIO supports geo scopes at US State and County granularities. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations. The data contain sentinel values (-999999) which
    represent values redacted for the sake of protecting patient privacy -- there
    were between 1 and 3 cases reported by the facility on that date.

    Parameters
    ----------
    fix_redacted : Fix[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to replace redacted values (-999999 in the data).
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data).
    """  # noqa: E501

    _ADULTS = "total_adult_patients_hospitalized_confirmed_covid_7_day_sum"
    _PEDS = "total_pediatric_patients_hospitalized_confirmed_covid_7_day_sum"

    _age_group: Literal["adult", "pediatric", "both"]

    def __init__(
        self,
        *,
        age_group: Literal["adult", "pediatric", "both"] = "both",
        fix_redacted: Fix[np.int64] | int | Callable[[], int] | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        if age_group not in ("adult", "pediatric", "both"):
            raise ValueError(f"Unsupported `age_group`: {age_group}")
        self._age_group = age_group
        super().__init__(fix_redacted=fix_redacted, fix_missing=fix_missing)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        match self._age_group:
            case "adult":
                values = [q.Select(self._ADULTS, "int", as_name="value")]
                not_nulls = [q.NotNull(self._ADULTS)]
            case "pediatric":
                values = [q.Select(self._PEDS, "int", as_name="value")]
                not_nulls = [q.NotNull(self._PEDS)]
            case "both":
                values = [
                    q.Select(self._ADULTS, "nullable_int", as_name="value_adult"),
                    q.Select(self._PEDS, "nullable_int", as_name="value_ped"),
                ]
                not_nulls = []
            case x:
                raise ValueError(f"Unsupported `age_group`: {x}")

        query = q.Query(
            select=(
                q.Select("collection_week", "date", as_name="date"),
                q.Select("fips_code", "str", as_name="geoid"),
                *values,
            ),
            where=q.And(
                q.DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("fips_code", context.scope.node_ids),
                *not_nulls,
            ),
            order_by=(
                q.Ascending("collection_week"),
                q.Ascending("fips_code"),
                q.Ascending(":id"),
            ),
        )
        try:
            return q.query_csv(
                resource=_HealthdataAnagCw7u._RESOURCE,
                query=query,
                api_token=healthdata_api_key(),
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame).to_numpy()
        processor = partial(
            process_txn,
            sentinels=[
                FixSentinel(
                    "redacted",
                    self._REDACTED_VALUE,
                    self._fix_redacted,
                ),
            ],
            fix_missing=self._fix_missing,
            dtype=np.int64,
            context=context,
            # data_df unspecified
            time_series=time_series,
        )

        if self._age_group == "both":
            # Age group is both adult and pediatric, so
            # process the two columns separately and sum.
            adult_df = (
                data_df[["date", "geoid", "value_adult"]]
                .rename(columns={"value_adult": "value"})
                .dropna(subset="value")
            )
            adult_df["value"] = adult_df["value"].astype(np.int64)
            adult_result = processor(data_df=adult_df)

            ped_df = (
                data_df[["date", "geoid", "value_ped"]]
                .rename(columns={"value_ped": "value"})
                .dropna(subset="value")
            )
            ped_df["value"] = ped_df["value"].astype(np.int64)
            ped_result = processor(data_df=ped_df)

            result = ProcessResult.sum(
                adult_result,
                ped_result,
                left_prefix="adult_",
                right_prefix="pediatric_",
            )
        else:
            # Age group is just adult or pediatric, so process as normal
            result = processor(data_df=data_df)
        return result.to_date_value(time_series)


class InfluenzaFacilityHospitalization(_HealthdataAnagCw7u):
    """
    Loads influenza hospitalization data from HealthData.gov's
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"
    dataset. The data were reported by healthcare facilities on a weekly basis,
    starting 2019-12-29 and ending 2024-04-21, although the data is not complete
    over this entire range, nor over the entire United States.

    This ADRIO supports geo scopes at US State and County granularities. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations. The data contain sentinel values (-999999) which
    represent values redacted for the sake of protecting patient privacy -- there
    were between 1 and 3 cases reported by the facility on that date.

    Parameters
    ----------
    fix_redacted : Fix[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to replace redacted values (-999999 in the data).
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data).
    """  # noqa: E501

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        query = q.Query(
            select=(
                q.Select("collection_week", "date", as_name="date"),
                q.Select("fips_code", "str", as_name="geoid"),
                q.Select(
                    "total_patients_hospitalized_confirmed_influenza_7_day_sum",
                    dtype="int",
                    as_name="value",
                ),
            ),
            where=q.And(
                q.DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("fips_code", context.scope.node_ids),
            ),
            order_by=(
                q.Ascending("collection_week"),
                q.Ascending("fips_code"),
                q.Ascending(":id"),
            ),
        )
        try:
            return q.query_csv(
                resource=_HealthdataAnagCw7u._RESOURCE,
                query=query,
                api_token=healthdata_api_key(),
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame).to_numpy()
        result = process_txn(
            sentinels=[
                FixSentinel(
                    "redacted",
                    self._REDACTED_VALUE,
                    self._fix_redacted,
                ),
            ],
            fix_missing=self._fix_missing,
            dtype=np.int64,
            context=context,
            data_df=data_df,
            time_series=time_series,
        )
        return result.to_date_value(time_series)


##########################
# DATA.CDC.GOV 3nnm-4jni #
##########################


class COVIDCountyCases(FetchADRIO[DateValueType, np.int64]):
    """
    Loads COVID case data from data.cdc.gov's dataset named
    "United States COVID-19 Community Levels by County".

    The data were reported starting 2022-02-24 and ending 2023-05-11, and aggregated
    by CDC to the US County level.

    This ADRIO supports geo scopes at US State and County granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    cases, calculated by multiplying the per-100k rates by the county population and rounding
    (via banker's rounding).

    Parameters
    ----------
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="3nnm-4jni")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2022-02-24"), iso8601("2023-05-11"), step=7)
    """The time range over which values are available."""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.AxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=True,
    )
    """The format of results."""

    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        *,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return self._RESULT_FORMAT

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        counties = cast(CensusScope, context.scope).as_granularity("county")
        query = q.Query(
            select=(
                q.Select("date_updated", "date", as_name="date"),
                q.Select("county_fips", "str", as_name="geoid"),
                q.SelectExpression(
                    "`covid_cases_per_100k` * (`county_population` / 100000)",
                    "float",
                    as_name="value",
                ),
            ),
            where=q.And(
                q.DateBetween(
                    "date_updated",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("county_fips", counties.node_ids),
            ),
            order_by=(
                q.Ascending("date_updated"),
                q.Ascending("county_fips"),
                q.Ascending(":id"),
            ),
        )
        try:
            return q.query_csv(
                resource=self._RESOURCE,
                query=query,
                api_token=data_cdc_api_key(),
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        if isinstance(context.scope, StateScope):
            data_df["geoid"] = data_df["geoid"].apply(STATE.truncate)
            data_df = data_df.groupby(["date", "geoid"]).sum().reset_index()
        data_df["value"] = np.round(data_df["value"])

        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame).to_numpy()
        result = process_txn(
            sentinels=[],
            fix_missing=self._fix_missing,
            dtype=np.int64,
            context=context,
            data_df=data_df,
            time_series=time_series,
        )
        return result.to_date_value(time_series)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


##########################
# DATA.CDC.GOV aemt-mg7g #
##########################
# TODO: is it worth keeping these states ADRIOs, or are the facility ADRIOs sufficient?
# (given that we can roll it up to state)
# compare results from each and see how far off they are, and if
# the time frame and geo availability are comparable.


class _DataCDCAemtMg7g(FetchADRIO[DateValueType, np.int64]):
    """
    An abstract specialization of `FetchADRIO` for ADRIOs which fetch
    data from cdc.gov dataset aemt-mg7g: a.k.a.
    "Weekly United States Hospitalization Metrics by Jurisdiction, During Mandatory
    Reporting Period from August 1, 2020 to April 30, 2024, and for Data Reported
    Voluntarily Beginning May 1, 2024, National Healthcare Safety Network
    (NHSN) - ARCHIVED".
    https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data
    """

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="aemt-mg7g")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2020-08-08"), iso8601("2024-10-26"), step=7)
    """The time range over which values are available."""

    _VOLUNTARY_TIME_RANGE = DateRange(
        iso8601("2024-05-04"), iso8601("2024-10-26"), step=7
    )
    """The time range over which values were reported voluntarily."""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.AxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=True,
    )
    """The format of results."""

    _column: str
    """The name of the data source column to fetch for this ADRIO."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""
    _allow_voluntary: bool

    def __init__(
        self,
        *,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
        allow_voluntary: bool = True,
    ):
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        self._allow_voluntary = allow_voluntary

    @property
    @override
    def result_format(self) -> ResultFormat[np.int64]:
        return self._RESULT_FORMAT

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope):
            err = "US State geo scope required."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        scope = cast(StateScope, self.context.scope)
        state_info = get_states(scope.year)
        to_postal = state_info.state_fips_to_code
        to_fips = state_info.state_code_to_fips

        query = q.Query(
            select=(
                q.Select("week_end_date", "date", as_name="date"),
                q.Select("jurisdiction", "str", as_name="geoid"),
                q.Select(self._column, "int", as_name="value"),
            ),
            where=q.And(
                q.DateBetween(
                    "week_end_date",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("jurisdiction", [to_postal[x] for x in scope.node_ids]),
                q.NotNull(self._column),
            ),
            order_by=(
                q.Ascending("week_end_date"),
                q.Ascending("jurisdiction"),
                q.Ascending(":id"),
            ),
        )
        try:
            result_df = q.query_csv(
                resource=self._RESOURCE,
                query=query,
                api_token=data_cdc_api_key(),
            )
            result_df["geoid"] = result_df["geoid"].apply(lambda x: to_fips[x])
            return result_df
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame).to_numpy()
        result = process_txn(
            sentinels=[],
            fix_missing=self._fix_missing,
            dtype=np.int64,
            context=context,
            data_df=data_df,
            time_series=time_series,
        )

        # If we don't allow voluntary reported data and if the time series overlaps
        # the voluntary period, add a data issue with the voluntary dates masked.
        vtr = self._VOLUNTARY_TIME_RANGE
        if (
            not self._allow_voluntary
            and (voluntary := vtr.overlap(context.time_frame)) is not None
        ):
            mask = np.isin(time_series, voluntary.to_numpy(), assume_unique=True)
            mask = np.broadcast_to(
                mask[:, np.newaxis],
                shape=(len(time_series), context.scope.nodes),
            )
            result = dataclasses.replace(
                result,
                issues=[*result.issues, ("voluntary", mask)],
            )

        return result.to_date_value(time_series)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_range = self._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


class COVIDStateHospitalization(_DataCDCAemtMg7g):
    """
    Loads COVID hospitalization data from data.cdc.gov's dataset named
    "Weekly United States Hospitalization Metrics by Jurisdiction, During Mandatory
    Reporting Period from August 1, 2020 to April 30, 2024, and for Data Reported
    Voluntarily Beginning May 1, 2024, National Healthcare Safety Network
    (NHSN) - ARCHIVED".

    The data were reported by healthcare facilities on a weekly basis to CDC's
    National Healthcare Safety Network with reporting dates starting 2020-08-08
    and ending 2024-10-26. The data were aggregated by CDC to the US State level.
    While reporting was initially federally required, beginning May 2024
    reporting became entirely voluntary and as such may include fewer responses.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations.

    Parameters
    ----------
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.
    allow_voluntary : bool, default=True
        Whether or not to accept voluntary data. If False and if the simulation time
        frame overlaps the voluntary period, such data will be masked.
        Set this to False if you want to be sure you are only using data during the
        required reporting period.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data).
    """  # noqa: E501

    _column = "total_admissions_all_covid_confirmed"


class FluStateHospitalization(_DataCDCAemtMg7g):
    """
    Loads influenza hospitalization data from data.cdc.gov's dataset named
    "Weekly United States Hospitalization Metrics by Jurisdiction, During Mandatory
    Reporting Period from August 1, 2020 to April 30, 2024, and for Data Reported
    Voluntarily Beginning May 1, 2024, National Healthcare Safety Network
    (NHSN) - ARCHIVED".

    The data were reported by healthcare facilities on a weekly basis to CDC's
    National Healthcare Safety Network with reporting dates starting 2020-08-08
    and ending 2024-10-26. The data were aggregated by CDC to the US State level.
    While reporting was initially federally required, beginning May 2024
    reporting became entirely voluntary and as such may include fewer responses.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations.

    Parameters
    ----------
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.
    allow_voluntary : bool, default=True
        Whether or not to accept voluntary data. If False and if the simulation time
        frame overlaps the voluntary period, such data will be masked.
        Set this to False if you want to be sure you are only using data during the
        required reporting period.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data).
    """  # noqa: E501

    _column = "total_admissions_all_influenza_confirmed"
