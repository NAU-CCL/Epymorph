"""
A geo specification contains metadata about a geo:
its attributes and specific dimensions in time and space.
"""
import calendar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from types import MappingProxyType
from typing import NamedTuple, Self, cast

import jsonpickle
import numpy as np
from numpy.typing import NDArray

from epymorph.data_shape import DataShape, Shapes
from epymorph.error import GeoValidationException
from epymorph.util import DTLike, NumpyTypeError, check_ndarray

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])
"""Structured numpy dtype for long/lat coordinates."""


class AttribDef(NamedTuple):
    """Metadata about a Geo attribute."""
    name: str
    dtype: DTLike
    shape: DataShape


LABEL = AttribDef('label', np.str_, Shapes.N)
"""
Label is a required attribute of every geo.
It is the source of truth for how many nodes are in the geo.
"""


class Geography(ABC):
    """
    Describes the geographic extent of a dynamic geo.
    Exactly how this extent is specified depends strongly on the data source.
    """


class TimePeriod(ABC):
    """Expresses the time period covered by a GeoSpec."""

    @property
    @abstractmethod
    def days(self) -> int:
        """The time period as a number of days."""


@dataclass(frozen=True)
class Year(TimePeriod):
    """TimePeriod representing a specific year."""
    year: int

    @property
    def days(self) -> int:
        return 366 if calendar.isleap(self.year) else 365


@dataclass(frozen=True)
class NonspecificDuration(TimePeriod):
    """
    TimePeriod representing a number of days not otherwise fixed in real time.
    This may be useful for testing purposes.
    """
    duration_days: int

    def __post_init__(self):
        if self.duration_days < 1:
            raise ValueError("duration_days must be at least 1.")

    @property
    def days(self) -> int:
        return self.duration_days


NO_DURATION = NonspecificDuration(1)


@dataclass(frozen=True)
class DateAndDuration(NonspecificDuration):
    """TimePeriod representing a number of days starting on the given date."""
    start_date: date

    @property
    def last_date(self) -> date:
        """
        Returns the last date included, i.e., the inclusive end of the date range.
        [start_date, last_date]
        """
        return self.start_date + timedelta(days=self.days-1)

    @property
    def end_date(self) -> date:
        """
        Returns the date after the last included date, i.e., the non-inclusive end of the date range.
        [start_date, end_date)
        """
        return self.start_date + timedelta(days=self.days)


@dataclass
class GeoSpec(ABC):
    """
    Abstract class describing a Geo.
    Subclasses will add fields and behavior specific to different types of Geos.
    """

    @classmethod
    def deserialize(cls, spec_string: str) -> Self:
        """deserializes a GEOSpec object from a pickled text"""
        spec = jsonpickle.decode(spec_string)
        if not isinstance(spec, cls):
            raise GeoValidationException('Invalid geo spec.')
        return spec

    attributes: list[AttribDef]
    """The attributes in the spec."""

    time_period: TimePeriod
    """
    The time period covered by the spec. By defining the time period,
    we can make reasonable assertions about whether any time-series data
    is well-formed.
    """

    @cached_property
    def attribute_map(self) -> MappingProxyType[str, AttribDef]:
        """The attributes in the spec, mapped by attribute name."""
        return MappingProxyType({a.name: a for a in self.attributes})

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'attribute_map' in state.keys():
            del state['attribute_map']  # don't pickle properties!
        return state

    def serialize(self) -> str:
        """Serializes this spec to string."""
        return cast(str, jsonpickle.encode(self, unpicklable=True))


@dataclass
class StaticGeoSpec(GeoSpec):
    """The spec for a StaticGeo."""
    sha256: str | None = field(default=None)
    """
    Optional: the sha256 checksum (hex encoded) for the data.npz file contents.
    If present, it will be used to verify the integrity of the data file when loaded.
    It will be added automatically when a StaticGeo is saved to a file.
    """


@dataclass
class DynamicGeoSpec(GeoSpec):
    """The spec for a DynamicGeo."""
    geography: Geography
    source: dict[str, str]


def validate_geo_values(spec: GeoSpec, values: dict[str, NDArray]) -> None:
    """
    Validate a set of geo values against the given GeoSpec.
    All spec'd attributes should be present and have the correct type and shape.
    Raises GeoValidationException for any errors.
    """
    if LABEL not in spec.attributes or LABEL.name not in values:
        msg = "Geo spec and values must both include the 'label' attribute."
        raise GeoValidationException(msg)

    N = len(values[LABEL.name])
    T = spec.time_period.days

    attribute_errors = list[str]()
    for a in spec.attributes:
        try:
            v = values[a.name]
            expected_shape = a.shape.as_tuple(N, T)
            check_ndarray(v, dtype=a.dtype, shape=expected_shape)
        except KeyError:
            msg = f"Geo is missing values for attribute '{a.name}'."
            attribute_errors.append(msg)
        except NumpyTypeError as e:
            msg = f"Geo attribute '{a.name}' is invalid. {e}"
            attribute_errors.append(msg)

    if len(attribute_errors) > 0:
        msg = "Geo contained invalid attributes."
        raise GeoValidationException(msg, attribute_errors)
