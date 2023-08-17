"""The core classes of the IPM system."""

from __future__ import annotations

from abc import ABC, abstractmethod

from epymorph.clock import Tick
from epymorph.context import Events, SimContext
from epymorph.movement.world import Location


class IpmBuilder(ABC):
    """Superclass for a class that builds IPMs bound to a context."""
    compartments: int
    events: int

    def __init__(self, num_compartments: int, num_events: int):
        self.compartments = num_compartments
        self.events = num_events

    def compartment_tags(self) -> list[list[str]]:
        """The tags associated with each copmartment in this model."""
        # Default value is no tag for any compartment.
        return [list() for _ in range(self.compartments)]

    @abstractmethod
    def verify(self, ctx: SimContext) -> None:
        """Verify whether or not this IPM has access to the data it needs."""

    @abstractmethod
    def build(self, ctx: SimContext) -> Ipm:
        """Construct an IPM for the given context."""


class Ipm(ABC):
    """Superclass for an Intra-population Model,
    used to calculate transition events at each tau step."""
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    @abstractmethod
    def events(self, loc: Location, tick: Tick) -> Events:
        """Calculate the events which took place in this tau step at the given location."""

    @abstractmethod
    def apply_events(self, loc: Location, es: Events) -> None:
        """Distribute events `es` among all populations at this location. (Modifies `loc`.)"""
