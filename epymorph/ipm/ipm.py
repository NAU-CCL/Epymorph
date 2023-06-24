
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.util import Compartments, Events
from epymorph.world import Location


class IpmBuilder(ABC):
    """
    Abstract class for a Builder of IPMs.
    The builder is constructed before we have a SimContext, but provides info that the SimContext needs.
    Later when we have a complete SimContext, this builder builds the IPM instance.
    """
    compartments: int
    events: int

    def __init__(self, num_compartments: int, num_events: int):
        self.compartments = num_compartments
        self.events = num_events

    def compartment_array(self) -> Compartments:
        """Build an empty compartment array of an appropriate size."""
        return np.zeros(self.compartments, dtype=np.int_)

    def event_array(self) -> Events:
        """Build an empty events array of an appropriate size."""
        return np.zeros(self.events, dtype=np.int_)

    @abstractmethod
    def verify(self, ctx: SimContext) -> None:
        """Verify whether or not this IPM has access to the data it needs."""
        pass

    @abstractmethod
    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        pass

    @abstractmethod
    def build(self, ctx: SimContext) -> Ipm:
        pass


class Ipm(ABC):
    """
    Abstract class for an Intra-Population Model,
    used to calculate transition events at each tau step.
    """
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    @abstractmethod
    def events(self, loc: Location,  tick: Tick) -> Events:
        """Calculate the events which took place in this tau step at the given location."""
        pass

    @abstractmethod
    def apply_events(self, loc: Location, es: Events) -> None:
        """Distribute events `es` among all populations at this location. (Modifies `loc`.)"""
        pass
