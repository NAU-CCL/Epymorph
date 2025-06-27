"""Data usage estimation and reporting."""

from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from math import floor, inf
from pathlib import Path
from shutil import disk_usage
from typing import Protocol, Sequence, runtime_checkable

from humanize import naturaldelta, naturalsize


@dataclass(frozen=True)
class EmptyDataEstimate:
    """
    When an entity is not capable of making a data usage estimate, it returns
    `EmptyDataEstimate` as a placeholder.

    Parameters
    ----------
    name :
        The name of the entity that provided the estimate.
    """

    name: str
    """The name of the entity that provided the estimate."""


@dataclass(frozen=True)
class AvailableDataEstimate:
    """
    An estimate for the data usage of a data fetch operation.

    Operations may download data and may utilize disk caching, so we would like
    to be able to estimate ahead of time how much data to expect. Because these are
    estimates, accuracy is not guaranteed.

    For example, an ADRIO which fetches data from a third-party source may be able to
    estimate ahead of time how much data needs to be downloaded and stored.

    Parameters
    ----------
    name :
        The name of the entity that provided the estimate.
    cache_key :
        A unique identifier for the data this estimate is about.
    new_network_bytes :
        How much new data (in bytes) will need to be downloaded.
    max_bandwidth :
        A source-specific limit on download bandwidth in bytes per second.
    new_cache_bytes :
        How much new data (in bytes) will be written to disk cache.
    total_cache_bytes :
        The total data (in bytes) that will be in the cache after fetch.
        This includes newly-cached and previously-cached files.
    """

    name: str
    """The name of the entity that provided the estimate."""
    cache_key: str
    """
    A unique identifier for the data this estimate is about.

    Multiple entities may load the same set of data; although both would report the same
    estimate, the actual data usage only happens for the first one to load. The rest
    would find and return the cached data. This key is used to distinguish this case --
    if two estimates share the same key, we can assume the estimate should only be
    counted once. Cache keys are only comparable within a single simulation context,
    so we don't need to perfectly distinguish between different scopes or time frames.
    """
    new_network_bytes: int
    """How much new data (in bytes) will need to be downloaded."""
    max_bandwidth: int | None
    """
    A source-specific limit on download bandwidth in bytes per second.
    (In case data sources impose known limits on download speed.)
    """
    new_cache_bytes: int
    """How much new data (in bytes) will be written to disk cache."""
    total_cache_bytes: int
    """
    The total data (in bytes) that will be in the cache after fetch.
    This includes newly-cached and previously-cached files.
    """


DataEstimate = EmptyDataEstimate | AvailableDataEstimate
"""`DataEstimate`s can be either empty or non-empty."""


@runtime_checkable
class CanEstimateData(Protocol):
    """A checkable protocol which indicates entities that can produce data estimates."""

    @abstractmethod
    def estimate_data(self) -> DataEstimate:
        """
        Estimate the data usage for this entity.

        If a reasonable estimate cannot be made, return an `EmptyDataEstimate`.

        Returns
        -------
        :
            The data estimate.
        """


@dataclass(frozen=True)
class DataEstimateTotal:
    """
    The computed total of one or more estimates.

    Parameters
    ----------
    new_network_bytes :
        How much new data (in bytes) will need to be downloaded.
    new_cache_bytes :
        How much new data (in bytes) will be written to disk cache.
    total_cache_bytes :
        The total data (in bytes) that will be in the cache after fetch.
    download_time :
        The estimated time (in seconds) to download all new data.
    """

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
    """
    Compute the total of a set of data estimates.

    A download time estimate is also provided, taking into account the assumed bandwidth
    limit (`max_bandwidth`) as well as any source-specific bandwidth limits.

    Parameters
    ----------
    estimates :
        The estimates to combine.
    max_bandwidth :
        The assumed maximum download bandwidth, in bytes per second.

    Returns
    -------
    :
        The estimate total.
    """
    new_net = 0
    new_cache = 0
    tot_cache = 0
    download_time = 0.0

    cache_keys = set[str]()
    for e in estimates:
        if not isinstance(e, AvailableDataEstimate):
            continue
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
    """
    Generate a report from the given set of data estimates.

    The report describes an itemized list of how much data will be downloaded and how
    much new data will be written to cache, then totals that up and reports how long
    that will take and whether or not there is enough available disk space.

    Parameters
    ----------
    cache_path :
        The path of epymorph's cache folder.
    estimates :
        The data estimates.
    max_bandwidth :
        The assumed maximum download bandwidth, in bytes per second.

    Returns
    -------
    :
        The report, as a list of lines.
    """
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
