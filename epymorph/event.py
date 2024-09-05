"""
epymorph's event frameworks.
The idea is to have a set of classes which define event protocols for
logical components of epymorph.
"""

from typing import NamedTuple

from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType
from epymorph.database import AbsoluteName
from epymorph.simulation import TimeFrame
from epymorph.util import Event, Singleton

#####################
# Simulation Events #
#####################


class OnStart(NamedTuple):
    """The payload of a simulation on_start event."""

    simulator: str
    """Name of the simulator class."""
    dim: SimDimensions
    """The dimensions of the simulation."""
    time_frame: TimeFrame
    """The timeframe for the simulation."""


class OnTick(NamedTuple):
    """The payload of a Simulation tick event."""

    tick_index: int
    percent_complete: float


###################
# Movement Events #
###################


class OnMovementStart(NamedTuple):
    """The payload for the event when movement processing starts for a tick."""

    tick: int
    """Which simulation tick."""
    day: int
    """Which simulation day."""
    step: int
    """Which tau step (by index)."""


class OnMovementClause(NamedTuple):
    """The payload for the event when a single movement clause has been processed."""

    tick: int
    """Which simulation tick."""
    day: int
    """Which simulation day."""
    step: int
    """Which tau step (by index)."""
    clause_name: str
    """The clause processed."""
    requested: NDArray[SimDType]
    """
    The number of individuals this clause 'wants' to move, that is, the values returned by its clause funcction.
    (An NxN array.)
    """
    actual: NDArray[SimDType]
    """The actual number of individuals moved, by source, destination, and compartment. (An NxNxC array.)"""
    total: int
    """The number of individuals moved by this clause."""
    is_throttled: bool
    """Did the clause request to move more people than were available (at any location)?"""


class OnMovementFinish(NamedTuple):
    """The payload for the event when movement processing finishes for one simulation tick."""

    tick: int
    """Which simulation tick."""
    day: int
    """Which simulation day."""
    step: int
    """Which tau step (by index)."""
    total: int
    """The total number of individuals moved during this tick."""


################
# ADRIO Events #
################


class AdrioStart(NamedTuple):
    """The payload of AdrioEvents.on_adrio_start"""

    adrio_name: str
    """The name of the ADRIO."""
    attribute: AbsoluteName
    """The name of the attribute."""


class AdrioFinish(NamedTuple):
    """The payload of AdrioEvents.on_adrio_finish"""

    adrio_name: str
    """The name of the ADRIO."""
    attribute: AbsoluteName
    """The name of the attribute."""
    duration: float
    """The number of seconds spent fetching."""


############
# EventBus #
############


class EventBus(metaclass=Singleton):
    """The one-stop for epymorph events. This class uses the singleton pattern."""

    # Simulation Events
    on_start: Event[OnStart]
    """Event fires at the start of a simulation run."""

    on_tick: Event[OnTick]
    """Event fires after each tick has been processed."""

    on_finish: Event[None]
    """Event fires after a simulation run is complete."""

    # Movement Events
    on_movement_start: Event[OnMovementStart]
    """Event fires at the start of the movement processing phase for every simulation tick."""

    on_movement_clause: Event[OnMovementClause]
    """Event fires after processing each active movement clause."""

    on_movement_finish: Event[OnMovementFinish]
    """Event fires at the end of the movement processing phase for every simulation tick."""

    # ADRIO Events
    on_adrio_start: Event[AdrioStart]
    """Event fires when an ADRIO is fetching data."""

    # on_adrio_progress: Event[AdrioProgress]
    # """Event that fires when..."""

    on_adrio_finish: Event[AdrioFinish]
    """Event fires when an ADRIO has finished fetching data."""

    def __init__(self):
        # SimulationEvents
        self.on_start = Event()
        self.on_tick = Event()
        self.on_finish = Event()
        # MovementEvents
        self.on_movement_start = Event()
        self.on_movement_clause = Event()
        self.on_movement_finish = Event()
        # AdrioEvents
        self.on_adrio_start = Event()
        self.on_adrio_finish = Event()
