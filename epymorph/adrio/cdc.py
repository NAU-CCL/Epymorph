from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Sequence, final, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.adrio.soda import (
    And,
    Ascending,
    DateBetween,
    In,
    Select,
    SocrataResource,
    query_csv,
)
from epymorph.attribute import AbsoluteName
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import Dimensions
from epymorph.database import DataResolver
from epymorph.error import DataResourceError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.time import DateRange, TimeFrame, iso8601


@dataclass(frozen=True)
class Context:
    name: AbsoluteName
    data: DataResolver
    scope: GeoScope
    time_frame: TimeFrame
    ipm: BaseCompartmentModel
    rng: np.random.Generator
    dim: Dimensions


@dataclass(frozen=True)
class Data:
    raw: pd.DataFrame
    issues: Sequence[tuple[str, NDArray[np.bool_]]]


DataTransformer = Callable[[Context, Data], Data]


class ADRIOPrototype(ABC):
    _transformers: Sequence[DataTransformer]

    def __init__(self, transformers: Sequence[DataTransformer]):
        self._transformers = transformers

    @abstractmethod
    def _validate(self, context: Context) -> None:
        pass

    @abstractmethod
    def _fetch(self, context: Context) -> Data:
        pass

    @final
    def _transform(self, context: Context, raw: Data) -> Data:
        return reduce(lambda x, t: t(context, x), self._transformers, raw)

    @abstractmethod
    def _format(self, context: Context, data: Data) -> np.ndarray:
        pass

    def evaluate(self, context: Context) -> np.ndarray:
        self._validate(context)
        raw = self._fetch(context)
        data = self._transform(context, raw)
        return self._format(context, data)


class InfluenzaFacilityHospitalization(ADRIOPrototype):
    _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), 7)

    def __init__(self, transformers: Sequence[DataTransformer]):
        super().__init__(transformers)

    @override
    def _validate(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise DataResourceError(err)

        start = context.time_frame.start_date
        end = context.time_frame.end_date
        if start < self._TIME_RANGE.start_date or end >= self._TIME_RANGE.stop_date:
            err = (
                "Facility-level hospitalization data is only available between "
                "12/29/2019 and 4/21/2024."
            )
            raise DataResourceError(err)

    @override
    def _fetch(self, context: Context) -> Data:
        res = SocrataResource(domain="healthdata.gov", id="anag-cw7u")
        value_col = "total_patients_hospitalized_confirmed_influenza_7_day_sum"

        result_df = query_csv(
            resource=res,
            select=(
                Select("collection_week", "date", as_name="date"),
                Select("fips_code", "str", as_name="geoid"),
                Select(value_col, "int", as_name="value"),
            ),
            where=And(
                In("fips_code", context.scope.node_ids),
                DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
            ),
            order_by=(
                Ascending("collection_week"),
                Ascending("fips_code"),
                Ascending(":id"),
            ),
        )

        expected_time_frame = pd.date_range("2019-12-29", "2024-04-21", freq="1W")

        raw_df = result_df.pivot_table(
            index="date",
            columns="geoid",
            values="value",
            aggfunc=lambda xs: -999999 if (xs == -999999).any() else xs.sum(),
            fill_value=0,
        ).reindex(
            # index=expected_time_frame, # TODO
            columns=context.scope.node_ids,
            fill_value=0,
        )

        missing_mask = (
            result_df.assign(value=False)
            .pivot_table(
                index="date",
                columns="geoid",
                values="value",
                aggfunc=lambda _: False,
                fill_value=True,
            )
            .reindex(columns=context.scope.node_ids, fill_value=True)
        ).to_numpy(dtype=np.bool_)

        return Data(raw_df, [("missing", missing_mask)])


# def query(
#     url_base: str,
#     location_col: str,
#     date_col: str,
#     value_col: str,
#     locations: list[str],
#     start_date: date,
#     end_date: date,
#     *,
#     and_where: list[str] | None = None,
#     limit: int = 10000,
#     parse_dates: bool = True,
# ) -> pd.DataFrame:
#     select_clause = ",".join([date_col, location_col, value_col])
#     location_list = ",".join(f"'{x}'" for x in locations)
#     location_clause = f"{location_col} IN ({location_list})"
#     date_clause = (
#         f"{date_col} BETWEEN '{start_date}T00:00:00' AND '{end_date}T00:00:00'"
#     )
#     where_clause = " AND ".join([location_clause, date_clause, *(and_where or [])])

#     def query_frames(offset: int = 0) -> Generator[pd.DataFrame, None, None]:
#         url = url_base + urlencode(
#             quote_via=quote,
#             safe=",()'$:",
#             query={
#                 "$select": select_clause,
#                 "$where": where_clause,
#                 "$limit": limit,
#                 "$offset": offset,
#             },
#         )

#         frame_df = pd.read_csv(url, dtype=str, parse_dates=parse_dates)
#         yield frame_df

#         if (frame_size := len(frame_df.index)) >= limit:
#             yield from query_frames(offset + frame_size)

#     return (
#         pd.concat(query_frames())
#         .rename(
#             columns={
#                 location_col: "geoid",
#                 date_col: "date",
#                 value_col: "value",
#             }
#         )
#         .sort_values(by=["date", "geoid"], ignore_index=True)
#     )
