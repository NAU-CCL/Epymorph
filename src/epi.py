from abc import ABC, abstractmethod

import numpy as np

from clock import Tick
from util import Compartments, Events
from world import Location


class Ipm(ABC):
    """Superclass for an Intra-population Model,
    used to calculate transition events at each tau step."""

    def __init__(self, num_compartments: int, num_events: int):
        self.num_compartments = num_compartments
        self.num_events = num_events

    def c(self) -> Compartments:
        """Build an empty compartment array of an appropriate size."""
        return np.zeros(self.num_compartments, dtype=np.int_)

    def e(self) -> Events:
        """Build an empty events array of an appropriate size."""
        return np.zeros(self.num_events, dtype=np.int_)

    @abstractmethod
    def initialize(self, num_nodes: int) -> list[Compartments]: pass

    @abstractmethod
    def events(self, loc: Location, tau: np.double, tick: Tick) -> Events: pass

    @abstractmethod
    def apply_events(self, loc: Location, es: Events) -> None: pass
