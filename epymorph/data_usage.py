from dataclasses import dataclass
from functools import partial
from math import floor, inf
from pathlib import Path
from shutil import disk_usage
from typing import Sequence, Union

from humanize import naturaldelta, naturalsize


@dataclass(frozen=True)
class EmptyDataEstimate:
    """An empty data estimate given that the provided data does not support the
    calculation of the data usage of a data fetch operation."""

    name: str
    """The name of the given ADRIO."""


@dataclass(frozen=True)
class AvailableDataEstimate:
    """An estimate for the data usage of a data fetch operation.

    Operations may download data and may utilize disk caching, so we would like
    to be able to estimate ahead of time how much data to expect.
    A concrete example of such an operation are ADRIOs fetch data from a third-party
    source during the preparation of a RUME.
    NOTE: all values are estimated and their accuracy may vary.
    """

    name: str
    """What is responsible for loading this data?"""
    cache_key: str
    """Multiple things may in fact load the same set of data; even though both would
    report the same estimate for missing data, only the first one to load would really
    incur that cost. The others would then find the cached data waiting.
    This key should make it possible to discover this case -- if two estimates are
    produced with the same key, it can be assumed that the estimate should only
    be counted once. Cache keys are only comparable within a single simulation context,
    so we don't need to perfectly distinguish between different scopes or time frames.
    """
    new_network_bytes: int
    """How much new data (in bytes) will need to be downloaded."""
    max_bandwidth: int | None
    """A source-specific limit on download bandwidth (in bytes per second).
    (Some sources may impose known limits on downloads.)
    """
    new_cache_bytes: int
    """How much new data (in bytes) will be written to disk cache."""
    total_cache_bytes: int
    """The total data (in bytes) that will be in the cache after fetch.
    This includes new cached files and previously cached files."""


DataEstimate = Union[EmptyDataEstimate, AvailableDataEstimate]


@dataclass(frozen=True)
class DataEstimateTotal:
    new_network_bytes: int
    """How much new data (in bytes) will need to be downloaded."""
    new_cache_bytes: int
    """How much new data (in bytes) will be written to disk cache."""
    total_cache_bytes: int
    """The total data (in bytes) that will be in the cache after fetch."""
    download_time: float
    """The estimated time (in seconds) to download all new data."""


def estimate_total(
    estimates: Sequence[DataEstimate],
    max_bandwidth: int,
) -> DataEstimateTotal:
    """Combines a number of individual data estimates into a total.

    Includes a total download time with the assumed bandwidth limit
    as well as source-specific bandwidth limits.
    """
    new_net = 0
    new_cache = 0
    tot_cache = 0
    download_time = 0.0

    cache_keys = set[str]()
    for e in estimates:
        if isinstance(e, AvailableDataEstimate):
            if e.cache_key in cache_keys:
                continue
            cache_keys.add(e.cache_key)
            new_net += e.new_network_bytes
            new_cache += e.new_cache_bytes
            tot_cache += e.total_cache_bytes
            download_time += e.new_network_bytes / (
                min(max_bandwidth, e.max_bandwidth or inf)
            )

    return DataEstimateTotal(new_net, new_cache, tot_cache, download_time)


def estimate_report(
    cache_path: Path,
    estimates: Sequence[DataEstimate],
    max_bandwidth: int,
) -> list[str]:
    """Generate a report from the given set of data estimates.

    Describes an itemized list of how much data will be downloaded and
    how much new data will be written to cache, then totals that up
    and reports how long that will take and whether or not there is enough
    available disk space."""
    # short-hand formatting functions
    ff = partial(naturalsize, binary=False)  # format file size
    ft = naturaldelta  # format time duration

    cache_keys = set[str]()
    result = list[str]()
    for e in estimates:
        if isinstance(e, AvailableDataEstimate):
            if e.cache_key in cache_keys or (
                (e.new_network_bytes) == 0 or (e.new_cache_bytes) == 0
            ):
                line = f"- {e.name} will be pulled from cache"
            else:
                line = f"- {e.name} will download {ff(e.new_network_bytes)} of new data"
            cache_keys.add(e.cache_key)
        else:
            line = f"- {e.name} (no estimate available)"
        result.append(line)

    total = estimate_total(estimates, max_bandwidth)
    result.append("In total we will:")

    if total.new_network_bytes == 0:
        result.append("- Download no additional data")
    else:
        result.append(
            f"- Download {ff(total.new_network_bytes)}, "
            f"taking {ft(total.download_time)} "
            f"(assuming {ff(max_bandwidth)}/s)"
        )

    available_space = disk_usage(cache_path).free
    if total.new_cache_bytes == 0:
        result.append("- Write no new data to disk cache")
    elif total.new_cache_bytes < floor(available_space * 0.9):
        result.append(
            f"- Write {ff(total.new_cache_bytes)} to disk cache "
            f"(you have {ff(available_space)} free space)"
        )
    elif total.new_cache_bytes < available_space:
        result.append(f"- Write {ff(total.new_cache_bytes)} to disk cache")
        result.append(
            "WARNING: this is very close to exceeding available free space "
            f"of {ff(available_space)}!"
        )
    else:
        result.append(f"- Write {ff(total.new_cache_bytes)} to disk cache")
        result.append(
            f"ERROR: this exceeds available free space of {ff(available_space)}!"
        )

    return result
