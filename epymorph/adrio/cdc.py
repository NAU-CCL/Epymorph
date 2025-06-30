"""ADRIOs for access US Centers for Disease Control data."""

import dataclasses
import os
from abc import abstractmethod
from datetime import date, timedelta
from typing import Callable, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

import epymorph.adrio.soda as q
from epymorph.adrio.adrio import (
    ADRIOCommunicationError,
    ADRIOContextError,
    FetchADRIO,
    PipelineResult,
    adrio_cache,
    adrio_validate_pipe,
    validate_time_frame,
)
from epymorph.adrio.processing import (
    DataPipeline,
    DateValueType,
    Fill,
    FillLikeFloat,
    FillLikeInt,
    Fix,
    FixLikeInt,
    PivotAxis,
)
from epymorph.adrio.validation import (
    ResultFormat,
    on_date_values,
    validate_dtype,
    validate_numpy,
    validate_shape,
    validate_values_in_range,
)
from epymorph.data_shape import Shapes
from epymorph.error import GeographyError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope, CountyScope, StateScope
from epymorph.geography.us_geography import STATE
from epymorph.geography.us_tiger import get_states
from epymorph.simulation import Context
from epymorph.time import DateRange, iso8601
from epymorph.util import date_value_dtype


def healthdata_api_key() -> str | None:
    """
    Load the Socrata API key to use for healthdata.gov,
    as environment variable 'API_KEY__healthdata.gov'.

    Returns
    -------
    :
        The key or `None` if it is not set.
    """
    return os.environ.get("API_KEY__healthdata.gov", default=None)


def data_cdc_api_key() -> str | None:
    """
    Load the Socrata API key to use for data.cdc.gov,
    as environment variable 'API_KEY__data.cdc.gov'.

    Returns
    -------
    :
        The key or `None` if it is not set.
    """
    return os.environ.get("API_KEY__data.cdc.gov", default=None)


def _truncate_county_to_scope_fn(scope: GeoScope) -> Callable[[str], str] | None:
    if isinstance(scope, StateScope):
        return STATE.truncate
    if isinstance(scope, CountyScope):
        return None
    err = f"Unsupported scope: {scope}."
    raise GeographyError(err)


############################
# HEALTHDATA.GOV anag-cw7u #
############################


class _HealthdataAnagCw7uMixin(FetchADRIO[DateValueType, np.int64]):
    """
    A mixin implementing some of `FetchADRIO`'s API for ADRIOs which fetch
    data from healthdata.gov dataset anag-cw7u: a.k.a.
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility".

    https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
    """

    _RESOURCE = q.SocrataResource(domain="healthdata.gov", id="anag-cw7u")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), step=7)
    """The time range over which values are available."""

    _REDACTED_VALUE = np.int64(-999999)
    """The value of redacted reports: between 1 and 3 cases."""

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        if context.scope.year != 2019:
            err = "This data supports 2019 Census geography only."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


@adrio_cache
class COVIDFacilityHospitalization(
    _HealthdataAnagCw7uMixin, FetchADRIO[DateValueType, np.int64]
):
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

    NOTE: this data source has a number of issues representing Alaska geography.
    It uses borough 02280 which isn't in the Census geography until 2020, and
    simultaneously uses pre-1980 Alaska geography (02080, 02120, 02210, and 02260).
    This makes these data inaccessible via this ADRIO. If Alaska data is important for
    your use-case, we recommend processing the data another way.

    Parameters
    ----------
    age_group :
        Which age group to fetch data for.
    fix_redacted :
        The method to use to replace redacted values (-999999 in the data).
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data).
    """  # noqa: E501

    _ADULTS = "total_adult_patients_hospitalized_confirmed_covid_7_day_sum"
    _PEDS = "total_pediatric_patients_hospitalized_confirmed_covid_7_day_sum"

    _age_group: Literal["adult", "pediatric", "both"]
    _fix_redacted: Fix[np.int64]
    _fix_missing: Fill[np.int64]

    def __init__(
        self,
        *,
        age_group: Literal["adult", "pediatric", "both"] = "both",
        fix_redacted: FixLikeInt = False,
        fix_missing: FillLikeInt = False,
    ):
        if age_group not in ("adult", "pediatric", "both"):
            raise ValueError(f"Unsupported `age_group`: {age_group}")
        self._age_group = age_group
        try:
            self._fix_redacted = Fix.of_int64(fix_redacted)
        except ValueError:
            raise ValueError("Invalid value for `fix_redacted`")
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

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
                resource=self._RESOURCE,
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
    ) -> PipelineResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.dtype["value"].type,
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

            result = PipelineResult.sum(
                adult_result,
                ped_result,
                left_prefix="adult_",
                right_prefix="pediatric_",
            )
        else:
            # Age group is just adult or pediatric, so process as normal
            result = pipeline(data_df)
        return result.to_date_value(time_series)


@adrio_cache
class InfluenzaFacilityHospitalization(
    _HealthdataAnagCw7uMixin, FetchADRIO[DateValueType, np.int64]
):
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

    NOTE: the data source has a number of issues representing Alaska geography.
    It uses borough 02280 which isn't in the Census geography until 2020, and
    simultaneously uses pre-1980 Alaska geography (02080, 02120, 02210, and 02260).
    This makes these data inaccessible via this ADRIO. If Alaska data is important for
    your use-case, we recommend processing the data another way.

    Parameters
    ----------
    fix_redacted :
        The method to use to replace redacted values (-999999 in the data).
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data).
    """  # noqa: E501

    _fix_redacted: Fix[np.int64]
    _fix_missing: Fill[np.int64]

    def __init__(
        self,
        *,
        fix_redacted: FixLikeInt = False,
        fix_missing: FillLikeInt = False,
    ):
        try:
            self._fix_redacted = Fix.of_int64(fix_redacted)
        except ValueError:
            raise ValueError("Invalid value for `fix_redacted`")
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

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
                resource=self._RESOURCE,
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
    ) -> PipelineResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.dtype["value"].type,
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


@adrio_cache
class COVIDCountyCases(FetchADRIO[DateValueType, np.int64]):
    """
    Loads COVID case data from data.cdc.gov's dataset named
    "United States COVID-19 Community Levels by County".

    The data were reported starting 2022-02-24 and ending 2023-05-11, and aggregated
    by CDC to the US County level.

    This ADRIO supports geo scopes at US State and County granularity (2015 through 2019
    allowed). The data loaded will be matched to the simulation time frame. The result
    is a 2D matrix where the first axis represents reporting weeks during the time frame
    and the second axis is geo scope nodes. Values are tuples of date and the integer
    number of cases, calculated by multiplying the per-100k rates by the county
    population and rounding (via banker's rounding).

    Parameters
    ----------
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="3nnm-4jni")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2022-02-24"), iso8601("2023-05-11"), step=7)
    """The time range over which values are available."""

    _fix_missing: Fill[np.int64]

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
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        if context.scope.year < 2015 or context.scope.year > 2019:
            err = "This data supports Census geography from 2015 through 2019 only."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        counties = cast(CensusScope, context.scope).as_granularity("county")

        # API URL can get too big if we're querying for a large number of counties.
        # And doing a very large "in" statement is not ideal anyway.
        # So after 1000 counties, just get all geography and we'll filter locally.
        if counties.nodes < 1000:
            geo_clause = [q.In("county_fips", counties.node_ids)]
            result_filter = None
        else:
            geo_clause = []
            result_filter = lambda df: df[df["geoid"].isin(counties.node_ids)]  # noqa: E731

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
                *geo_clause,
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
                result_filter=result_filter,
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> PipelineResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.dtype["value"].type,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=_truncate_county_to_scope_fn(context.scope),
            )
            .map_series("value", map_fn=lambda xs: xs.round())
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


##########################
# DATA.CDC.GOV aemt-mg7g #
##########################


class _DataCDCAemtMg7gMixin(FetchADRIO[DateValueType, np.int64]):
    """
    An mixin implemeting some of `FetchADRIO`'s API for ADRIOs which fetch
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

    _column: str
    """The name of the data source column to fetch for this ADRIO."""
    _fix_missing: Fill[np.int64]
    """The method to use to fix missing values."""
    _allow_voluntary: bool
    """Whether or not to accept voluntary data."""

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope):
            err = "US State geo scope required."
            raise ADRIOContextError(self, context, err)
        # No year restriction since state-equivalents are the same
        # for the entire supported time range.
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
    ) -> PipelineResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = DataPipeline(
            axes=(
                PivotAxis("date", time_series),
                PivotAxis("geoid", context.scope.node_ids),
            ),
            ndims=2,
            dtype=self.result_format.dtype["value"].type,
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
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


@adrio_cache
class COVIDStateHospitalization(
    _DataCDCAemtMg7gMixin, FetchADRIO[DateValueType, np.int64]
):
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
    fix_missing :
        The method to use to fix missing values.
    allow_voluntary :
        Whether or not to accept voluntary data. If False and if the simulation time
        frame overlaps the voluntary period, such data will be masked.
        Set this to False if you want to be sure you are only using data during the
        required reporting period.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data).
    """  # noqa: E501

    _column = "total_admissions_all_covid_confirmed"

    def __init__(
        self,
        *,
        fix_missing: FillLikeInt = False,
        allow_voluntary: bool = True,
    ):
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        self._allow_voluntary = allow_voluntary


@adrio_cache
class InfluenzaStateHospitalization(
    _DataCDCAemtMg7gMixin, FetchADRIO[DateValueType, np.int64]
):
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
    fix_missing :
        The method to use to fix missing values.
    allow_voluntary :
        Whether or not to accept voluntary data. If False and if the simulation time
        frame overlaps the voluntary period, such data will be masked.
        Set this to False if you want to be sure you are only using data during the
        required reporting period.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data).
    """  # noqa: E501

    _column = "total_admissions_all_influenza_confirmed"

    def __init__(
        self,
        *,
        fix_missing: FillLikeInt = False,
        allow_voluntary: bool = True,
    ):
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")
        self._allow_voluntary = allow_voluntary


##########################
# DATA.CDC.GOV 8xkx-amqh #
##########################


@adrio_cache
class COVIDVaccination(FetchADRIO[DateValueType, np.int64]):
    """
    Loads COVID hospitalization data from data.cdc.gov's dataset named
    "COVID-19 Vaccinations in the United States,County".

    The data cover a time period starting 2020-12-13 and ending 2023-05-10.
    Up through 2022-06-16, data were reported on a daily cadence, and after
    that switched to a weekly cadence.

    This ADRIO supports geo scopes at US State and County granularity (2015 through 2019
    allowed). The data appears to have been compiled using 2019 Census delineations, so
    for best results, use a geo scope for that year. The data loaded will be matched
    to the simulation time frame. The result is a 2D matrix where the first axis
    represents reporting dates during the time frame and the second axis is geo scope
    nodes. Values are tuples of date and the integer number of people who have had the
    requested vaccine dosage.

    Parameters
    ----------
    vaccine_status :
        The dataset breaks down vaccination status by how many doses individuals have
        received. Use this to specify which status you're interested in.
        "at least one dose" includes people who have received at least one COVID vaccine
        dose; "full series" includes people who have received at least either
        two doses of a two-dose vaccine or one dose of a one-dose vaccine;
        "full series and booster" includes people who have received the full series
        and at least one booster dose.
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="8xkx-amqh")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2020-12-13"), iso8601("2023-05-10"), step=1)
    """The time range over which values are available."""
    _DAILY_TIME_RANGE = DateRange(iso8601("2020-12-13"), iso8601("2022-06-16"), step=1)
    """The time range during which data were reported daily."""
    _WEEKLY_TIME_RANGE = DateRange(iso8601("2022-06-22"), iso8601("2023-05-10"), step=7)
    """The time range during which data were reported weekly."""

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
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        if context.scope.year < 2015 or context.scope.year > 2019:
            err = "This data supports Census geography from 2015 through 2019 only."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    def _calculate_time_series(self) -> NDArray[np.datetime64]:
        time_frame = self.context.time_frame
        daily = self._DAILY_TIME_RANGE.overlap(time_frame)
        weekly = self._WEEKLY_TIME_RANGE.overlap(time_frame)
        match (daily, weekly):
            case None, None:
                # this should be caught during validate_context(), but just in case...
                err = "The supplied time frame does not include any available dates."
                raise ADRIOContextError(self, self.context, err)
            case d, None:
                return d.to_numpy()
            case None, w:
                return w.to_numpy()
            case d, w:
                return np.concatenate((d.to_numpy(), w.to_numpy()))

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        # There are three cases to handle:
        # - StateScope
        # - CountyScope with # nodes <= 1000
        # - CountyScope with # nodes > 1000
        #   In this case API URL can get too big, so just get all geography
        #   and we'll filter locally.
        if isinstance(context.scope, StateScope):
            to_postal = get_states(context.scope.year).state_fips_to_code
            postal_codes = [to_postal[x] for x in context.scope.node_ids]
            geo_clause = [q.In("Recip_State", postal_codes)]
            result_filter = None
        elif isinstance(context.scope, CountyScope) and context.scope.nodes < 1000:
            geo_clause = [q.In("fips", context.scope.node_ids)]
            result_filter = None
        elif isinstance(context.scope, CountyScope):
            geo_clause = []
            result_filter = lambda df: df[df["geoid"].isin(context.scope.node_ids)]  # noqa: E731
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
                *geo_clause,
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
                result_filter=result_filter,
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> PipelineResult[DateValueType]:
        time_series = self._calculate_time_series()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=self.result_format.dtype["value"].type,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=_truncate_county_to_scope_fn(context.scope),
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._calculate_time_series()
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


##########################
# DATA.CDC.GOV ite7-j2w7 #
##########################


@adrio_cache
class CountyDeaths(FetchADRIO[DateValueType, np.int64]):
    """
    Loads COVID and total deaths data from data.cdc.gov's dataset named
    "AH COVID-19 Death Counts by County and Week, 2020-present".

    The data were reported starting 2020-01-04 and ending 2023-04-01, and aggregated
    by CDC to the US County level.

    This ADRIO supports geo scopes at US State and County granularity (2014 through 2019
    allowed). The data loaded will be matched to the simulation time frame. The result
    is a 2D matrix where the first axis represents reporting weeks during the time frame
    and the second axis is geo scope nodes. Values are tuples of date and the integer
    number of deaths.

    NOTE: this data source uses non-standard geography for two county-equivalents.
    In Alaska, 02270 was the Wade Hampton Census Area prior to 2015 and thereafter
    renamed Kusilvak Census Area with code 02158. And in South Dakota, 46113 was
    Shannon County prior to 2015 and thereafter renamed Oglala Lakota County
    with code 46102. These data are inaccessible via this ADRIO unless you use
    2014 geography.

    Parameters
    ----------
    cause_of_death :
        The cause of death.
    fix_redacted :
        The method to use to fix redacted values.
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="ite7-j2w7")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange(iso8601("2020-01-04"), iso8601("2023-04-01"), step=7)
    """The time range over which values are available."""

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
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)
        if context.scope.year < 2014 or context.scope.year > 2019:
            err = "This data supports Census geography from 2014 through 2019 only."
            raise ADRIOContextError(self, context, err)
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        # Data represents county-level FIPS codes as numbers,
        # so we strip leading zeros when querying and
        # left-pad with zero to get back to five characters in the result.
        counties = cast(CensusScope, context.scope).as_granularity("county")

        if counties.nodes < 1000:
            geo_clause = [q.In("fips_code", counties.node_ids)]

            def result_filter(df):
                df["geoid"] = df["geoid"].str.rjust(5, "0")
                return df
        else:
            geo_clause = []

            def result_filter(df):
                df["geoid"] = df["geoid"].str.rjust(5, "0")
                return df[df["geoid"].isin(counties.node_ids)]

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
                *geo_clause,
            ),
            order_by=(
                q.Ascending("week_ending_date"),
                q.Ascending("fips_code"),
                q.Ascending(":id"),
            ),
        )
        try:
            return q.query_csv(
                resource=self._RESOURCE,
                query=query,
                api_token=data_cdc_api_key(),
                result_filter=result_filter,
            )
        except Exception as e:
            raise ADRIOCommunicationError(self, context) from e

    @override
    def _process(
        self,
        context: Context,
        data_df: pd.DataFrame,
    ) -> PipelineResult[DateValueType]:
        value_dtype = self.result_format.dtype["value"].type
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=value_dtype,
                rng=context,
            )
            .map_column(
                "geoid",
                map_fn=_truncate_county_to_scope_fn(context.scope),
            )
            # insert our own sentinel value: nulls in these data mean "redacted"
            # but we can't have null values for the finalize step
            .strip_na_as_sentinel(
                "value",
                sentinel_name="redacted",
                sentinel_value=value_dtype(-999999),
                fix=self._fix_redacted,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


##########################
# DATA.CDC.GOV r8kw-7aab #
##########################


@adrio_cache
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
    cause_of_death :
        The cause of death.
    fix_redacted :
        The method to use to fix redacted values.
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data).
    """  # noqa: E501

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="r8kw-7aab")
    """The Socrata API endpoint."""

    @staticmethod
    def _time_range() -> DateRange:
        """Compute the time range over which values are available."""
        # There's about a one week lag in the data.
        # On a Thursday they seem to post data up to the previous Saturday,
        # so this config handles that.
        return DateRange.until_date(
            iso8601("2020-01-04"),
            date.today() - timedelta(days=7),
            step=7,
        )

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
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope):
            err = "US State geo scope required."
            raise ADRIOContextError(self, context, err)
        # No year restriction since state-equivalents are the same
        # for the entire supported time range.
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
    ) -> PipelineResult[DateValueType]:
        value_dtype = self.result_format.dtype["value"].type
        time_series = self._time_range().overlap_or_raise(context.time_frame).to_numpy()
        pipeline = (
            DataPipeline(
                axes=(
                    PivotAxis("date", time_series),
                    PivotAxis("geoid", context.scope.node_ids),
                ),
                ndims=2,
                dtype=value_dtype,
                rng=context,
            )
            # insert our own sentinel value: nulls in these data mean "redacted"
            # but we can't have null values for the finalize step
            .strip_na_as_sentinel(
                "value",
                sentinel_name="redacted",
                sentinel_value=value_dtype(-999999),
                fix=self._fix_redacted,
            )
            .finalize(self._fix_missing)
        )
        return pipeline(data_df).to_date_value(time_series)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._time_range().overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


##########################
# DATA.CDC.GOV mpgq-jmmr #
##########################

ResultT = TypeVar("ResultT", bound=np.generic)


class _DataCDCMpgqJmmrMixin(FetchADRIO[DateValueType, ResultT]):
    """
    An mixin implemeting some of `FetchADRIO`'s API for ADRIOs which fetch
    data from cdc.gov dataset mpgq-jmmr: a.k.a.
    "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction,
    National Healthcare Safety Network (NHSN) (Preliminary)".

    https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data
    """

    _RESOURCE = q.SocrataResource(domain="data.cdc.gov", id="mpgq-jmmr")
    """The Socrata API endpoint."""

    _TIME_RANGE = DateRange.until_date(
        start_date=iso8601("2024-11-02"),
        max_end_date=date.today() - timedelta(days=1),
        step=7,
    )
    """The time range over which values are available."""

    _fix_missing: Fill[ResultT]
    """The method to use to fix missing values."""

    @property
    @abstractmethod
    def _column_name(self) -> str:
        """The column to fetch from the data source."""

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope):
            err = "US State geo scope required."
            raise ADRIOContextError(self, context, err)
        # No geo year restriction since state-equivalents are the same
        # for the entire supported time range.
        validate_time_frame(self, context, self._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        scope = cast(StateScope, self.context.scope)
        state_info = get_states(scope.year)
        to_postal = state_info.state_fips_to_code
        to_fips = state_info.state_code_to_fips

        val_dtype = self.result_format.dtype["value"]
        col_type = "float" if val_dtype == np.float64 else "int"

        query = q.Query(
            select=(
                q.Select("weekendingdate", "date", as_name="date"),
                q.Select("jurisdiction", "str", as_name="geoid"),
                q.Select(self._column_name, col_type, as_name="value"),
            ),
            where=q.And(
                q.DateBetween(
                    "weekendingdate",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("jurisdiction", [to_postal[x] for x in scope.node_ids]),
                q.NotNull(self._column_name),
            ),
            order_by=(
                q.Ascending("weekendingdate"),
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
    ) -> PipelineResult[DateValueType]:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame).to_numpy()
        pipeline = DataPipeline(
            axes=(
                PivotAxis("date", time_series),
                PivotAxis("geoid", context.scope.node_ids),
            ),
            ndims=2,
            dtype=self.result_format.dtype["value"].type,
            rng=context,
        ).finalize(self._fix_missing)
        result = pipeline(data_df)
        return result.to_date_value(time_series)

    @override
    def validate_result(self, context: Context, result: NDArray) -> None:
        time_series = self._TIME_RANGE.overlap_or_raise(context.time_frame)
        result_shape = (len(time_series), context.scope.nodes)
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(result_shape),
            validate_dtype(self.result_format.dtype),
            on_date_values(validate_values_in_range(0, None)),
        )


@adrio_cache
class CurrentStateHospitalization(
    _DataCDCMpgqJmmrMixin[np.int64], FetchADRIO[DateValueType, np.int64]
):
    """
    Loads disease-specific hospitalization data from data.cdc.gov's dataset named
    "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction,
    National Healthcare Safety Network (NHSN) (Preliminary)".

    The data are currently being reported by healthcare facilities on a weekly basis to
    CDC's National Healthcare Safety Network. This ADRIO loads recent data from the
    source, starting 2024-11-02 and onward. (Data exist before this date but with some
    caveats; see `COVIDStateHospitalization` and `InfluenzaStateHospitalization` for
    ADRIOs which support earlier, archived data.) The data were aggregated by CDC to
    the US State level.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Dates represent the MMWR week ending date of the
    data collection week. Values are tuples of date and the integer number of reported
    hospitalizations.

    Parameters
    ----------
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data).
    [epymorph.adrio.cdc.COVIDStateHospitalization][] and
    [epymorph.adrio.cdc.InfluenzaStateHospitalization][] for data prior to 2024-11-01.
    """  # noqa: E501

    Disease = Literal["Covid", "Influenza", "RSV"]
    """A disease category available in this data."""
    AgeGroup = Literal["Total", "Adult", "Pediatric"]
    """An age category available in this data."""

    _DISEASE_VALUES = {
        "Covid": "c19",
        "Influenza": "flu",
        "RSV": "rsv",
    }

    # column name format is primarily dependent on the chosen age group,
    # and then we insert the disease identifier
    _AGE_GROUP_VALUES = {
        "Total": "totalconf{}hosppats",
        "Adult": "numconf{}hosppatsadult",
        "Pediatric": "numconf{}hosppatsped",
    }

    _disease: Disease
    """The disease to load."""
    _age_group: AgeGroup
    """The age group to load."""

    def __init__(
        self,
        *,
        disease: Disease,
        age_group: AgeGroup = "Total",
        fix_missing: FillLikeInt = False,
    ):
        if disease not in self._DISEASE_VALUES:
            err = f"Not a supported disease type: {disease}"
            raise ValueError(err)
        if age_group not in self._AGE_GROUP_VALUES:
            err = f"Not a supported age group type: {age_group}"
            raise ValueError(err)
        self._disease = disease
        self._age_group = age_group
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def _column_name(self) -> str:
        disease = self._DISEASE_VALUES[self._disease]
        column_template = self._AGE_GROUP_VALUES[self._age_group]
        return column_template.format(disease)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))


@adrio_cache
class CurrentStateHospitalizationICU(
    _DataCDCMpgqJmmrMixin[np.float64], FetchADRIO[DateValueType, np.float64]
):
    """
    Loads disease-specific ICU hospitalization data from data.cdc.gov's dataset named
    "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction,
    National Healthcare Safety Network (NHSN) (Preliminary)".

    The data are currently being reported by healthcare facilities on a weekly basis to
    CDC's National Healthcare Safety Network. This ADRIO loads recent data from the
    source, starting 2024-11-02 and onward. The data were aggregated by CDC to
    the US State level.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Dates represent the MMWR week ending date of the
    data collection week. Values are tuples of date and the integer number of reported
    ICU hospitalizations.

    Parameters
    ----------
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data).
    """  # noqa: E501

    Disease = Literal["Covid", "Influenza", "RSV"]
    """A disease category available in this data."""
    AgeGroup = Literal["Total", "Adult", "Pediatric"]
    """An age category available in this data."""

    _DISEASE_VALUES = {
        "Covid": "c19",
        "Influenza": "flu",
        "RSV": "rsv",
    }

    # column name format is primarily dependent on the chosen age group,
    # and then we insert the disease identifier
    _AGE_GROUP_VALUES = {
        "Total": "totalconf{}icupats",
        "Adult": "numconf{}icupatsadult",
        "Pediatric": "numconf{}icupatsped",
    }

    _disease: Disease
    """The disease to load."""
    _age_group: AgeGroup
    """The age group to load."""

    def __init__(
        self,
        *,
        disease: Disease,
        age_group: AgeGroup = "Total",
        fix_missing: FillLikeFloat = False,
    ):
        if disease not in self._DISEASE_VALUES:
            err = f"Not a supported disease type: {disease}"
            raise ValueError(err)
        if age_group not in self._AGE_GROUP_VALUES:
            err = f"Not a supported age group type: {age_group}"
            raise ValueError(err)
        self._disease = disease
        self._age_group = age_group
        try:
            self._fix_missing = Fill.of_float64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def _column_name(self) -> str:
        disease = self._DISEASE_VALUES[self._disease]
        column_template = self._AGE_GROUP_VALUES[self._age_group]
        return column_template.format(disease)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.float64))


@adrio_cache
class CurrentStateAdmissions(
    _DataCDCMpgqJmmrMixin[np.int64], FetchADRIO[DateValueType, np.int64]
):
    """
    Loads disease-specific hospital admissions data from data.cdc.gov's dataset named
    "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction,
    National Healthcare Safety Network (NHSN) (Preliminary)".

    The data are currently being reported by healthcare facilities on a weekly basis to
    CDC's National Healthcare Safety Network. This ADRIO loads recent data from the
    source, starting 2024-11-02 and onward. (Data exist before this date but with some
    caveats; see `COVIDStateHospitalization` and `InfluenzaStateHospitalization` for
    ADRIOs which support earlier, archived data.) The data were aggregated by CDC to
    the US State level.

    This ADRIO supports geo scopes at US State granularity. The data
    loaded will be matched to the simulation time frame. The result is a 2D matrix
    where the first axis represents reporting weeks during the time frame and the
    second axis is geo scope nodes. Dates represent the MMWR week ending date of the
    data collection week. Values are tuples of date and the integer number of reported
    hospital admissions.

    Parameters
    ----------
    fix_missing :
        The method to use to fix missing values.

    See Also
    --------
    [The dataset documentation](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data).
    [epymorph.adrio.cdc.COVIDStateHospitalization][] and
    [epymorph.adrio.cdc.InfluenzaStateHospitalization][] for data prior to 2024-11-01.
    """  # noqa: E501

    Disease = Literal["Covid", "Influenza", "RSV"]
    """A disease category available in this data."""
    AgeGroup = Literal[
        "0 to 4",
        "5 to 17",
        "18 to 49",
        "50 to 64",
        "65 to 74",
        "75 and above",
        "Unknown",
        "Adult",
        "Pediatric",
        "Total",
    ]
    """An age category available in this data."""

    _DISEASE_VALUES = {
        "Covid": "c19",
        "Influenza": "flu",
        "RSV": "rsv",
    }

    # column name format is primarily dependent on the chosen age group,
    # and then we insert the disease identifier
    _AGE_GROUP_VALUES = {
        "0 to 4": "numconf{}newadmped0to4",
        "5 to 17": "numconf{}newadmped5to17",
        "18 to 49": "numconf{}newadmadult18to49",
        "50 to 64": "numconf{}newadmadult50to64",
        "65 to 74": "numconf{}newadmadult65to74",
        "75 and above": "numconf{}newadmadult75plus",
        "Unknown": "numconf{}newadmunk",
        "Adult": "totalconf{}newadmadult",
        "Pediatric": "totalconf{}newadmped",
        "Total": "totalconf{}newadm",
    }

    _disease: Disease
    """The disease to load."""
    _age_group: AgeGroup
    """The age group to load."""

    def __init__(
        self,
        *,
        disease: Disease,
        age_group: AgeGroup = "Total",
        fix_missing: FillLikeInt = False,
    ):
        if disease not in self._DISEASE_VALUES:
            err = f"Not a supported disease type: {disease}"
            raise ValueError(err)
        if age_group not in self._AGE_GROUP_VALUES:
            err = f"Not a supported age group type: {age_group}"
            raise ValueError(err)
        self._disease = disease
        self._age_group = age_group
        try:
            self._fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @property
    @override
    def _column_name(self) -> str:
        disease = self._DISEASE_VALUES[self._disease]
        column_template = self._AGE_GROUP_VALUES[self._age_group]
        return column_template.format(disease)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.AxN, dtype=date_value_dtype(np.int64))
