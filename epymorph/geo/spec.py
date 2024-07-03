"""
A geo specification contains metadata about a geo:
its attributes and specific dimensions in time and space.
"""
import calendar
from abc import ABC
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from types import MappingProxyType
from typing import Any, Self, cast

import jsonpickle
from numpy.typing import NDArray

import epymorph.data_shape as shape
from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geography.scope import GeoScope
from epymorph.simulation import AttributeDef
from epymorph.util import NumpyTypeError, check_ndarray, match

LABEL = AttributeDef('label', type=str, shape=Shapes.N,
                     comment='The label associated with each node.')
"""
Label is a required attribute of every geo.
It is the source of truth for how many nodes are in the geo.
"""


class Geography(ABC):
    """
    Describes the geographic extent of a dynamic geo.
    Exactly how this extent is specified depends strongly on the data source.
    """


@dataclass(frozen=True)
class TimePeriod(ABC):
    """Expresses the time period covered by a GeoSpec."""

    days: int
    """The time period as a number of days."""


@dataclass(frozen=True)
class SpecificTimePeriod(TimePeriod, ABC):
    """Expresses a real time period, with a determinable start and end date."""

    start_date: date
    """The start date of the date range. [start_date, end_date)"""
    end_date: date
    """The non-inclusive end date of the date range. [start_date, end_date)"""


@dataclass(frozen=True)
class DateRange(SpecificTimePeriod):
    """TimePeriod representing the time between two dates, exclusive of the end date."""
    start_date: date
    end_date: date
    days: int = field(init=False)

    def __post_init__(self):
        days = (self.end_date - self.start_date).days
        object.__setattr__(self, 'days', days)


@dataclass(frozen=True)
class DateAndDuration(SpecificTimePeriod):
    """TimePeriod representing a number of days starting on the given date."""
    days: int
    start_date: date
    end_date: date = field(init=False)

    def __post_init__(self):
        end_date = self.start_date + timedelta(days=self.days)
        object.__setattr__(self, 'end_date', end_date)


@dataclass(frozen=True)
class Year(SpecificTimePeriod):
    """TimePeriod representing a specific year."""
    year: int
    days: int = field(init=False)
    start_date: date = field(init=False)
    end_date: date = field(init=False)

    def __post_init__(self):
        days = 366 if calendar.isleap(self.year) else 365
        start_date = date(self.year, 1, 1)
        end_date = date(self.year + 1, 1, 1)
        object.__setattr__(self, 'days', days)
        object.__setattr__(self, 'start_date', start_date)
        object.__setattr__(self, 'end_date', end_date)


@dataclass(frozen=True)
class NonspecificDuration(TimePeriod):
    """
    TimePeriod representing a number of days not otherwise fixed in real time.
    This may be useful for testing purposes.
    """

    def __post_init__(self):
        if self.days < 1:
            raise ValueError("duration_days must be at least 1.")


NO_DURATION = NonspecificDuration(1)


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

    attributes: list[AttributeDef]
    """The attributes in the spec."""

    scope: GeoScope
    """
    The physical bounds of this geo: how many nodes are included?
    Under some geographic systems (like the US Census delineations),
    this may include the hierarchical granularity of the nodes, and
    which delineation year we're using.
    """

    time_period: TimePeriod
    """
    The time period covered by the spec. By defining the time period,
    we can make reasonable assertions about whether any time-series data
    is well-formed.
    """

    @cached_property
    def attribute_map(self) -> MappingProxyType[str, AttributeDef]:
        """The attributes in the spec, mapped by attribute name."""
        return MappingProxyType({a.name: a for a in self.attributes})

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'attribute_map' in state:
            del state['attribute_map']  # don't pickle properties!
        return state

    def serialize(self) -> str:
        """Serializes this spec to string."""
        return cast(str, jsonpickle.encode(self, unpicklable=True))


@dataclass
class StaticGeoSpec(GeoSpec):
    """The spec for a StaticGeo."""
    # Nothing but the default stuff here.


@dataclass
class DynamicGeoSpec(GeoSpec):
    """The spec for a DynamicGeo."""
    source: dict[str, Any]


def validate_geo_values(spec: GeoSpec, values: dict[str, NDArray]) -> None:
    """
    Validate a set of geo values against the given GeoSpec.
    All spec'd attributes should be present and have the correct type and shape.
    Raises GeoValidationException for any errors.
    """
    # TODO: this isn't being called anymore by the BasicSimulator (aka StandardSim).
    # But I'll leave it here until we rip out Geos entirely.

    if LABEL not in spec.attributes or LABEL.name not in values:
        msg = "Geo spec and values must both include the 'label' attribute."
        raise GeoValidationException(msg)

    N = len(values[LABEL.name])
    T = spec.time_period.days

    attribute_errors = list[str]()
    for a in spec.attributes:
        try:
            value = values[a.name]
            check_ndarray(value, dtype=match.dtype_cast(a.dtype))
            # check_ndarray's shape matching requires SimDimensions (which we don't have)
            # So fake its logic for the time being.
            match a.shape:
                case shape.Time():
                    shape_matches = value.shape == (T,)
                case shape.Node():
                    shape_matches = value.shape == (N,)
                case shape.TimeAndNode():
                    shape_matches = value.shape == (T, N)
                case shape.NodeAndNode():
                    shape_matches = value.shape == (N, N)
                case _:
                    msg = f"Geo attribute is using an unsupported shape: {a.name}; {a.shape}"
                    raise GeoValidationException(msg)
            if not shape_matches:
                msg = f"Not a numpy shape match: got {value.shape}, expected {a.shape}"
                raise NumpyTypeError(msg)
        except KeyError:
            msg = f"Geo is missing values for attribute '{a.name}'."
            attribute_errors.append(msg)
        except NumpyTypeError as e:
            msg = f"Geo attribute '{a.name}' is invalid. {e}"
            attribute_errors.append(msg)

    if len(attribute_errors) > 0:
        msg = "Geo contained invalid attributes."
        raise GeoValidationException(msg, attribute_errors)
