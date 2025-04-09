import dataclasses
import os
from datetime import date, timedelta
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
    DataPipeline,
    DateValueType,
    Fill,
    Fix,
    PivotAxis,
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
        if context.scope.year != 2019:
            err = "This data supports 2019 Census geography only."
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


class COVIDFacilityHospitalization(_HealthdataAnagCw7u):
    """
    Loads COVID hospitalization data from HealthData.gov's
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"
    dataset. The data were reported by healthcare facilities on a weekly basis,
    starting 2019-12-29 and ending 2024-04-21, although the data is not complete
    over this entire range, nor over the entire United States.

    This ADRIO supports geo scopes at US State and County granularities in 2019.
    The data loaded will be matched to the simulation time frame. The result is a 2D
    matrix where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations. The data contain sentinel values (-999999) which
    represent values redacted for the sake of protecting patient privacy -- there
    were between 1 and 3 cases reported by the facility on that date.

    NOTE: the data has a number of issues representing Alaska geography.
    It uses borough 02280 which isn't in the Census geography until 2020, and
    it uses non-standard Alaska geography (02080, 02120, 02210, and 02260) which are not
    county-equivalents. This makes these data inaccessible via this ADRIO.
    If Alaska data is important for your use-case, we recommend processing the data
    another way.

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
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .strip_sentinel("redacted", self._REDACTED_VALUE, self._fix_redacted)
            .finalize(self._fix_missing)
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
            adult_result = pipeline(adult_df)

            ped_df = (
                data_df[["date", "geoid", "value_ped"]]
                .rename(columns={"value_ped": "value"})
                .dropna(subset="value")
            )
            ped_df["value"] = ped_df["value"].astype(np.int64)
            ped_result = pipeline(ped_df)

            result = ProcessResult.sum(
                adult_result,
                ped_result,
                left_prefix="adult_",
                right_prefix="pediatric_",
            )
        else:
            # Age group is just adult or pediatric, so process as normal
            result = pipeline(data_df)
        return result.to_date_value(time_series)


class InfluenzaFacilityHospitalization(_HealthdataAnagCw7u):
    """
    Loads influenza hospitalization data from HealthData.gov's
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"
    dataset. The data were reported by healthcare facilities on a weekly basis,
    starting 2019-12-29 and ending 2024-04-21, although the data is not complete
    over this entire range, nor over the entire United States.

    This ADRIO supports geo scopes at US State and County granularities in 2019.
    The data loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    reported hospitalizations. The data contain sentinel values (-999999) which
    represent values redacted for the sake of protecting patient privacy -- there
    were between 1 and 3 cases reported by the facility on that date.

    NOTE: the data has a number of issues representing Alaska geography.
    It uses borough 02280 which isn't in the Census geography until 2020, and
    it uses non-standard Alaska geography (02080, 02120, 02210, and 02260) which are not
    county-equivalents. This makes these data inaccessible via this ADRIO.
    If Alaska data is important for your use-case, we recommend processing the data
    another way.

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
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .strip_sentinel(
                "redacted",
                self._REDACTED_VALUE,
                self._fix_redacted,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)


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
        # TODO: which county years work?
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
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=STATE.truncate
                if isinstance(context.scope, StateScope)
                else None,
                dtype=np.str_,
            )
            .map_series("value", map_fn=lambda xs: xs.round())
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


##########################
# DATA.CDC.GOV aemt-mg7g #
##########################


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
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = DataPipeline(
            axes=(
                PivotAxis("date", time_series),
                PivotAxis("geoid", context.scope.node_ids),
            ),
            ndims=2,
            dtype=self.result_format.value_dtype,
            rng=context,
        ).finalize(self._fix_missing)
        result = pipeline(data_df)

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
                issues={**result.issues, "voluntary": mask},
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
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
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


class InfluenzaStateHospitalization(_DataCDCAemtMg7g):
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


##########################
# DATA.CDC.GOV 8xkx-amqh #
##########################


class COVIDVaccination(FetchADRIO[np.int64, np.int64]):
    """
    Loads COVID hospitalization data from data.cdc.gov's dataset named
    "COVID-19 Vaccinations in the United States,County".

    The data were reported on a daily basis, starting 2020-12-13 and ending 2023-05-10.

    This ADRIO supports geo scopes at US State and County granularity. The data appears
    to have been compiled using 2019 Census delineations, so for best results, use
    a geo scope for that year. The data loaded will be matched to the simulation time
    frame. The result is a 2D matrix where the first axis represents reporting dates
    during the time frame and the second axis is geo scope nodes. Values are integer
    numbers of people who have had the requested vaccine dosage.

    Parameters
    ----------
    vaccine_status : Literal["at least one dose", "full series", "full series and booster"]
        The dataset breaks down vaccination status by how many doses individuals have
        received. Use this to specify which status you're interested in.
        "at least one dose" includes people who have received at least one COVID vaccine
        dose; "full series" includes people who have received at least either
        two doses of a two-dose vaccine or one dose of a one-dose vaccine;
        "full series and booster" includes people who have received the full series
        and at least one booster dose.
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="8xkx-amqh")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2020-12-13"), iso8601("2023-05-10"), step=1)
    """The time range over which values are available."""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.TxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=False,
    )
    """The format of results."""

    _vaccine_status: Literal[
        "at least one dose", "full series", "full series and booster"
    ]
    """The datapoint to fetch for this ADRIO."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        vaccine_status: Literal[
            "at least one dose", "full series", "full series and booster"
        ],
        *,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        if vaccine_status not in (
            "at least one dose",
            "full series",
            "full series and booster",
        ):
            raise ValueError(f"Invalid value for `vaccine_status`: {vaccine_status}")
        self._vaccine_status = vaccine_status
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
        scope = context.scope
        if not isinstance(scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        # NOTE: Alaska geography is a minor issue.
        # 2019 has county 02261 which is gone in 2020, and
        # 2020 has counties 02063 and 02066 which aren't in the data.
        # I'm not making this an error though.
        # Using a scope with a mismatched year may result in more missing data,
        # so users can deal with it that way.
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        if isinstance(context.scope, StateScope):
            to_postal = get_states(context.scope.year).state_fips_to_code
            postal_codes = [to_postal[x] for x in context.scope.node_ids]
            loc_where = q.In("Recip_State", postal_codes)
        elif isinstance(context.scope, CountyScope):
            loc_where = q.In("fips", context.scope.node_ids)
        else:
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)

        match self._vaccine_status:
            case "at least one dose":
                column = "administered_dose1_recip"
            case "full series":
                column = "series_complete_yes"
            case "full series and booster":
                column = "booster_doses"

        query = q.Query(
            select=(
                q.Select("date", "date", as_name="date"),
                q.Select("fips", "str", as_name="geoid"),
                q.Select(column, "nullable_int", as_name="value"),
            ),
            where=q.And(
                q.DateBetween(
                    "date",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                loc_where,
                q.NotNull(column),
            ),
            order_by=(
                q.Ascending("date"),
                q.Ascending("fips"),
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
    ) -> ProcessResult[np.int64]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=STATE.truncate
                if isinstance(context.scope, StateScope)
                else None,
                dtype=np.str_,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df)


##########################
# DATA.CDC.GOV ite7-j2w7 #
##########################


class CountyDeaths(FetchADRIO[DateValueType, np.int64]):
    """
    Loads COVID and total deaths data from data.cdc.gov's dataset named
    "AH COVID-19 Death Counts by County and Week, 2020-present".

    The data were reported starting 2020-01-04 and ending 2023-04-01, and aggregated
    by CDC to the US County level.

    This ADRIO supports geo scopes at US State and County granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    deaths.

    Parameters
    ----------
    cause_of_death : Literal["all", "COVID-19"]
        The cause of death.
    fix_redacted : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix redacted values.
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="ite7-j2w7")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2020-01-04"), iso8601("2023-04-01"), step=7)
    """The time range over which values are available."""

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.AxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=True,
    )
    """The format of results."""

    _cause_of_death: Literal["all", "COVID-19"]
    """The cause of death."""
    _fix_redacted: Fix[np.int64]
    """The method to use to replace redacted values (N/A in the data)."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        cause_of_death: Literal["all", "COVID-19"],
        *,
        fix_redacted: Fix[np.int64] | int | Callable[[], int] | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        if cause_of_death not in ("all", "COVID-19"):
            err = f"Unsupported cause of death: {cause_of_death}"
            raise ValueError(err)
        self._cause_of_death = cause_of_death
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
    def _fetch(self, context: Context) -> pd.DataFrame:
        counties = cast(CensusScope, context.scope).as_granularity("county")
        # Data represents county-level FIPS codes as numbers,
        # so we have to strip leading zeros when querying (???) and
        # left-pad with zero to get back to five characters in the result.

        match self._cause_of_death:
            case "all":
                value_column = "total_deaths"
            case "COVID-19":
                value_column = "covid_19_deaths"

        query = q.Query(
            select=(
                q.Select("week_ending_date", "date", as_name="date"),
                q.Select("fips_code", "str", as_name="geoid"),
                q.SelectExpression(
                    value_column,
                    "nullable_int",
                    as_name="value",
                ),
            ),
            where=q.And(
                q.DateBetween(
                    "week_ending_date",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("fips_code", counties.node_ids),
            ),
            order_by=(
                q.Ascending("week_ending_date"),
                q.Ascending("fips_code"),
                q.Ascending(":id"),
            ),
        )
        try:
            result_df = q.query_csv(
                resource=self._RESOURCE,
                query=query,
                api_token=data_cdc_api_key(),
            )
            result_df["geoid"] = result_df["geoid"].str.rjust(5, "0")
            return result_df
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=STATE.truncate
                if isinstance(context.scope, StateScope)
                else None,
                dtype=np.str_,
            )
            # insert our own sentinel value: nulls in these data mean "redacted"
            # but we can't have null values for the finalize step
            .strip_na_as_sentinel(
                sentinel_name="redacted",
                sentinel_value=-999999,
                fix=self._fix_redacted,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)


##########################
# DATA.CDC.GOV r8kw-7aab #
##########################


class StateDeaths(FetchADRIO[DateValueType, np.int64]):
    """
    Loads deaths data (COVID-19, influenza, pneumonia, and total) from data.cdc.gov's dataset named
    "Provisional COVID-19 Death Counts by Week Ending Date and State".

    The data were reported starting 2020-01-04 and aggregated by CDC to the US State level.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Values are tuples of date and the integer number of
    deaths.

    Parameters
    ----------
    cause_of_death : Literal["all", "COVID-19", "influenza", "pneumonia"]
        The cause of death.
    fix_redacted : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix redacted values.
    fix_missing : Fill[np.int64] | int | Callable[[], int] | Literal[False], default=False
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="r8kw-7aab")
    """The Socrata API endpoint."""

    @staticmethod
    def _time_range() -> DateRange:
        """The time range over which values are available."""
        # There's about a one week lag in the data.
        # On a Thursday they seem to post data up to the previous Saturday,
        # so this config handles that.
        return DateRange.until_date(
            iso8601("2020-01-04"),
            date.today() - timedelta(days=7),
            step=7,
        )

    _RESULT_FORMAT = ResultFormat(
        shape=Shapes.AxN,
        value_dtype=np.int64,
        validation=range_mask_fn(minimum=np.int64(0), maximum=None),
        is_date_value=True,
    )
    """The format of results."""

    _cause_of_death: Literal["all", "COVID-19", "influenza", "pneumonia"]
    """The cause of death."""
    _fix_redacted: Fix[np.int64]
    """The method to use to replace redacted values (N/A in the data)."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""

    def __init__(
        self,
        cause_of_death: Literal["all", "COVID-19", "influenza", "pneumonia"],
        *,
        fix_redacted: Fix[np.int64] | int | Callable[[], int] | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        if cause_of_death not in ("all", "COVID-19", "influenza", "pneumonia"):
            err = f"Unsupported cause of death: {cause_of_death}"
            raise ValueError(err)
        self._cause_of_death = cause_of_death
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
        if not isinstance(context.scope, StateScope):
            err = "US State geo scope required."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._time_range())

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        # Data contains state names, so create mappings to and from.
        scope = cast(StateScope, context.scope)
        to_state = get_states(year=scope.year).state_fips_to_name
        to_fips = {to_state[x]: x for x in scope.node_ids}
        states = to_fips.keys()

        match self._cause_of_death:
            case "all":
                value_column = "total_deaths"
            case "COVID-19":
                value_column = "covid_19_deaths"
            case "influenza":
                value_column = "influenza_deaths"
            case "pneumonia":
                value_column = "pneumonia_deaths"

        query = q.Query(
            select=(
                q.Select("week_ending_date", "date", as_name="date"),
                q.Select("state", "str", as_name="geoid"),
                q.SelectExpression(
                    value_column,
                    "nullable_int",
                    as_name="value",
                ),
            ),
            where=q.And(
                q.Equals("group", "By Week"),
                q.DateBetween(
                    "week_ending_date",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("state", states),
            ),
            order_by=(
                q.Ascending("week_ending_date"),
                q.Ascending("state"),
                q.Ascending(":id"),
            ),
        )
        try:
            result_df = q.query_csv(
                resource=self._RESOURCE,
                query=query,
                api_token=data_cdc_api_key(),
            )
            geoid = result_df["geoid"].apply(lambda x: to_fips[x])
            return result_df.assign(geoid=geoid)
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> ProcessResult[DateValueType]:
        time_series = self._time_range().overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.value_dtype,
                rng=context,
            )
            # insert our own sentinel value: nulls in these data mean "redacted"
            # but we can't have null values for the finalize step
            .strip_na_as_sentinel(
                sentinel_name="redacted",
                sentinel_value=-999999,
                fix=self._fix_redacted,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_series = self._time_range().overlap_or_raise(context.time_frame)
        shp = (len(time_series), context.scope.nodes)
        super()._validate_result(context, result, expected_shape=shp)
