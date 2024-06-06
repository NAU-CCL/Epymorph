"""
A geo specification contains metadata about a geo:
its attributes and specific dimensions in time and space.
"""
import calendar
from abc import ABC
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from types import MappingProxyType
from typing import Self, cast

import jsonpickle
from numpy.typing import NDArray

from epymorph.data_shape import DataShape, Shapes
from epymorph.data_type import DataDType
from epymorph.error import GeoValidationException
from epymorph.geography.scope import GeoScope
from epymorph.simulation import AttributeDef, geo_attrib
from epymorph.sympy_shim import to_symbol
from epymorph.util import NumpyTypeError, check_ndarray

LABEL = geo_attrib('label', dtype=str, shape=Shapes.N,
                   comment='The label associated with each node.')
"""
Label is a required attribute of every geo.
It is the source of truth for how many nodes are in the geo.
"""


class SourceSpec(ABC):
    """
    Contains information needed by a data source to correctly fetch an attribute.
    Type of information varies by data source, as well as if any is needed at all.
    """


class Geography(ABC):
    """
    Describes the geographic extent of a dynamic geo.
    Exactly how this extent is specified depends strongly on the data source.
    """


class TimePeriod(ABC):
    """Expresses the time period covered by a GeoSpec."""

    days: int
    """The time period as a number of days."""


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

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days


@dataclass(frozen=True)
class Year(SpecificTimePeriod):
    """TimePeriod representing a specific year."""
    year: int

    @property
    def days(self) -> int:
        return 366 if calendar.isleap(self.year) else 365

    @property
    def start_date(self) -> date:
        return date(self.year, 1, 1)

    @property
    def end_date(self) -> date:
        return date(self.year + 1, 1, 1)


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

    attributes: list[AttributeDef]
    """The attributes in the spec."""

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
    scope: GeoScope
    source: dict[str, str | SourceSpec]


def attrib(name: str, dtype: DataDType, shape: DataShape = Shapes.N, comment: str | None = None):
    """
    A convenience constructor for an AttributeDef as used in the geo spec or an ADRIO maker.
    Although we use the same class for any context, there are certain arguments which aren't really
    useful when writing a geo spec or ADRIO maker, and so this method hides the non-useful options.
    """
    return AttributeDef(
        name=name,
        source='geo',
        dtype=dtype,
        shape=shape,
        symbol=to_symbol(name),
        default_value=None,
        comment=comment,
    )


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
            check_ndarray(v, dtype=[a.dtype], shape=expected_shape)
        except KeyError:
            msg = f"Geo is missing values for attribute '{a.name}'."
            attribute_errors.append(msg)
        except NumpyTypeError as e:
            msg = f"Geo attribute '{a.name}' is invalid. {e}"
            attribute_errors.append(msg)

    if len(attribute_errors) > 0:
        msg = "Geo contained invalid attributes."
        raise GeoValidationException(msg, attribute_errors)
