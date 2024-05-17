"""
epymorph's event frameworks.
The idea is to have a set of classes which define event protocols for 
logical components of epymorph.
"""
from typing import NamedTuple, Protocol, runtime_checkable

from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType
from epymorph.simulation import TimeFrame
from epymorph.util import Event

# Simulation Events


class OnStart(NamedTuple):
    """The payload of a Simulation on_start event."""
    dim: SimDimensions
    time_frame: TimeFrame


class OnTick(NamedTuple):
    """The payload of a Simulation tick event."""
    tick_index: int
    percent_complete: float


@runtime_checkable
class SimulationEvents(Protocol):
    """
    Protocol for Simulations that support lifecycle events.
    For correct operation, ensure that `on_start` is fired first,
    then `on_tick` any number of times, then finally `on_end`.
    """

    on_start: Event[OnStart]
    """
    Event fires at the start of a simulation run. Payload is a subset of the context for this run.
    """

    on_tick: Event[OnTick]
    """
    Event which fires after each tick has been processed.
    Event payload is a tuple containing the tick index just completed (an integer),
    and the percentage complete (a float).
    """

    # TODO: rename `on_end` to `on_finish`.

    on_end: Event[None]
    """
    Event fires after a simulation run is complete.
    """


class SimulationEventsMixin(SimulationEvents):
    """A mixin implementation of the SimulationEvents protocol which initializes the events."""

    def __init__(self):
        self.on_start = Event()
        self.on_tick = Event()
        self.on_end = Event()

    def has_subscribers(self) -> bool:
        """True if there is at least one subscriber on any simulation event."""
        return self.on_start.has_subscribers \
            or self.on_tick.has_subscribers \
            or self.on_end.has_subscribers


# Movement Events


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


@runtime_checkable
class MovementEvents(Protocol):
    """
    Mixin for Simulations that support movement events.
    For correct operation, ensure that `on_movement_start` is fired first,
    then `on_movement_clause` any number of times, then finally `on_movement_finish`.
    """

    on_movement_start: Event[OnMovementStart]
    """
    Event fires at the start of the movement processing phase for every simulation tick.
    """

    on_movement_clause: Event[OnMovementClause]
    """
    Event fires after every movement clause has been processed, excluding clauses that are not triggered in this tick.
    """

    on_movement_finish: Event[OnMovementFinish]
    """
    Event fires at the end of the movement processing phase for every simulation tick.
    """


class MovementEventsMixin(MovementEvents):
    """A mixin implementation of the MovementEvents protocol which initializes the events."""

    def __init__(self):
        self.on_movement_start = Event()
        self.on_movement_clause = Event()
        self.on_movement_finish = Event()

    def has_subscribers(self) -> bool:
        """True if there is at least one subscriber on any movement event."""
        return self.on_movement_start.has_subscribers \
            or self.on_movement_clause.has_subscribers \
            or self.on_movement_finish.has_subscribers


class SimWithEvents(SimulationEvents, MovementEvents, Protocol):
    """Intersection type of SimulationEvents and MovementEvents"""


# Geo/ADRIO Events


class FetchStart(NamedTuple):
    """The payload of a DynamicGeo fetch_start event."""
    adrio_len: int


class AdrioStart(NamedTuple):
    """The payload of a DynamicGeo adrio_start event."""
    attribute: str
    index: int | None
    """An index assigned to this ADRIO if fetching ADRIOs as a batch."""
    adrio_len: int | None
    """The total number of ADRIOs being fetched if fetching ADRIOs as a batch."""


@runtime_checkable
class DynamicGeoEvents(Protocol):
    """
    Protocol for DynamicGeos that support lifecycle events.
    For correct operation, ensure that `fetch_start` is fired first,
    then `adrio_start` any number of times, then finally `fetch_end`.
    """

    fetch_start: Event[FetchStart]
    """
    Event that fires when geo begins fetching attributes. Payload is the number of ADRIOs.
    """

    adrio_start: Event[AdrioStart]
    """
    Event that fires when an individual ADRIO begins data retreival. Payload is the attribute name and index.
    """

    fetch_end: Event[None]
    """
    Event that fires when data retreival is complete.
    """
