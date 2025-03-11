import os
from typing import Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

import epymorph.adrio.soda as q
from epymorph.adrio.adrio import (
    ADRIOPrototype,
    Fill,
    Fix,
    FixSentinel,
    ProcessResult,
    process_txn,
    range_mask_fn,
    validate_time_frame,
)
from epymorph.error import DataResourceError
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.simulation import Context
from epymorph.time import DateRange, iso8601


def healthdata_api_key() -> str | None:
    return os.environ.get("API_KEY__healthdata.gov", default=None)


class InfluenzaFacilityHospitalization(ADRIOPrototype[np.int64]):
    _RESOURCE = q.SocrataResource(domain="healthdata.gov", id="anag-cw7u")
    _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), step=7)
    _VALUE_RANGE = range_mask_fn(minimum=np.int64(0), maximum=None)
    _REDACTED_VALUE = np.int64(-999999)

    fix_redacted: Fix[np.int64]
    fix_missing: Fill[np.int64]

    def __init__(
        self,
        *,
        fix_redacted: Fix[np.int64] | int | Callable[[], int] | Literal[False] = False,
        fix_missing: Fill[np.int64] | int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            self.fix_redacted = Fix.of_int64(fix_redacted)
        except ValueError:
            raise ValueError("Invalid value for `fix_redacted`")
        try:
            self.fix_missing = Fill.of_int64(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @override
    def _validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise DataResourceError(err)

        validate_time_frame(
            InfluenzaFacilityHospitalization._TIME_RANGE,
            context.time_frame,
        )

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        value_col = "total_patients_hospitalized_confirmed_influenza_7_day_sum"
        query = q.Query(
            select=(
                q.Select("collection_week", "date", as_name="date"),
                q.Select("fips_code", "str", as_name="geoid"),
                q.Select(value_col, "int", as_name="value"),
            ),
            where=q.And(
                q.In("fips_code", context.scope.node_ids),
                q.DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
            ),
            order_by=(
                q.Ascending("collection_week"),
                q.Ascending("fips_code"),
                q.Ascending(":id"),
            ),
        )
        return q.query_csv(
            resource=InfluenzaFacilityHospitalization._RESOURCE,
            query=query,
            api_token=healthdata_api_key(),
        )

    @override
    def _process(self, context: Context, data_df: pd.DataFrame) -> ProcessResult:
        return process_txn(
            data_availability=InfluenzaFacilityHospitalization._TIME_RANGE,
            sentinels=[
                FixSentinel(
                    "redacted",
                    InfluenzaFacilityHospitalization._REDACTED_VALUE,
                    self.fix_redacted,
                )
            ],
            fix_missing=self.fix_missing,
            dtype=np.int64,
            context=context,
            data_df=data_df,
        )

    @override
    def _validate_result(self, context: Context, result: NDArray[np.int64]) -> None:
        # NOTE: validation only checks non-masked values
        is_valid = InfluenzaFacilityHospitalization._VALUE_RANGE
        if not is_valid(result).all():
            raise ValueError("invalid values")  # TODO

        time_series = InfluenzaFacilityHospitalization._TIME_RANGE.overlap(
            context.time_frame
        )
        if time_series is None:
            raise ValueError("this should really be caught ealier")  # TODO
        expected_shape = (len(time_series), context.scope.nodes)
        if result.shape != expected_shape:
            raise ValueError("invalid shape")  # TODO

        if np.dtype(result.dtype) != np.dtype(np.int64):
            raise ValueError("invalid dtype")  # TODO
