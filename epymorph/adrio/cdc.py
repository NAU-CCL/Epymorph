from datetime import date
from typing import Generator
from urllib.parse import quote, urlencode

import pandas as pd


def query(
    url_base: str,
    location_col: str,
    date_col: str,
    value_col: str,
    locations: list[str],
    start_date: date,
    end_date: date,
    *,
    and_where: list[str] | None = None,
    limit: int = 10000,
    parse_dates: bool = True,
) -> pd.DataFrame:
    select_clause = ",".join([date_col, location_col, value_col])
    location_list = ",".join(f"'{x}'" for x in locations)
    location_clause = f"{location_col} IN ({location_list})"
    date_clause = (
        f"{date_col} BETWEEN '{start_date}T00:00:00' AND '{end_date}T00:00:00'"
    )
    where_clause = " AND ".join([location_clause, date_clause, *(and_where or [])])

    def query_frames(offset: int = 0) -> Generator[pd.DataFrame, None, None]:
        url = url_base + urlencode(
            quote_via=quote,
            safe=",()'$:",
            query={
                "$select": select_clause,
                "$where": where_clause,
                "$limit": limit,
                "$offset": offset,
            },
        )

        frame_df = pd.read_csv(url, dtype=str, parse_dates=parse_dates)
        yield frame_df

        if (frame_size := len(frame_df.index)) >= limit:
            yield from query_frames(offset + frame_size)

    return (
        pd.concat(query_frames())
        .rename(
            columns={
                location_col: "geoid",
                date_col: "date",
                value_col: "value",
            }
        )
        .sort_values(by=["date", "geoid"], ignore_index=True)
    )
