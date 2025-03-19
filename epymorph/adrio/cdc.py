import os
from typing import Callable, Literal

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
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.simulation import Context
from epymorph.time import DateRange, iso8601


def healthdata_api_key() -> str | None:
    """
    Loads the Socrata API key to use for healthdata.gov,
    as environment variable 'API_KEY__healthdata.gov'.
    """
    return os.environ.get("API_KEY__healthdata.gov", default=None)


class _HealthdataAnagCw7u(FetchADRIO[DateValueType, np.int64]):
    """
    An abstract specialization of `FetchADRIO` for ADRIOs which fetch
    data from healthdata.gov dataset anag-cw7u: a.k.a.
    "COVID-19 Reported Patient Impact and Hospital Capacity by Facility".
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

    _column: str
    """The name of the data source column to fetch for this ADRIO."""
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
        return _HealthdataAnagCw7u._RESULT_FORMAT

    @override
    def validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise ADRIOContextError(self, context, err)

        validate_time_frame(self, context, _HealthdataAnagCw7u._TIME_RANGE)

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        query = q.Query(
            select=(
                q.Select("collection_week", "date", as_name="date"),
                q.Select("fips_code", "str", as_name="geoid"),
                q.Select(self._column, "int", as_name="value"),
            ),
            where=q.And(
                q.DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
                q.In("fips_code", context.scope.node_ids),
                q.NotNull(self._column),
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
        time_range = _HealthdataAnagCw7u._TIME_RANGE
        time_series = time_range.overlap_or_raise(context.time_frame).to_numpy()
        result = process_txn(
            sentinels=[
                FixSentinel(
                    "redacted",
                    _HealthdataAnagCw7u._REDACTED_VALUE,
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

    @override
    def _validate_result(
        self,
        context: Context,
        result: NDArray[DateValueType],
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        time_range = _HealthdataAnagCw7u._TIME_RANGE
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

    _column = "total_adult_patients_hospitalized_confirmed_covid_7_day_sum"

    # TODO: should we include pediatric hospitalized?


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

    _column = "total_patients_hospitalized_confirmed_influenza_7_day_sum"
