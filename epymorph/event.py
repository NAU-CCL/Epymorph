"""
epymorph's event frameworks.
The idea is to have a set of classes which define event protocols for
logical components of epymorph.
"""

from typing import NamedTuple

from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType
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
    The number of individuals this clause 'wants' to move, that is,
    the values returned by its clause function. (An NxN array.)
    """
    actual: NDArray[SimDType]
    """
    The actual number of individuals moved, by source, destination, and compartment.
    (An NxNxC array.)
    """
    total: int
    """The number of individuals moved by this clause."""
    is_throttled: bool
    """
    Did the clause request to move more people than were available (at any location)?
    """


class OnMovementFinish(NamedTuple):
    """
    The payload for the event when movement processing finishes for one simulation tick.
    """

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


class DownloadActivity(NamedTuple):
    """A description of ADRIO network download activity.

    All fields are optional, as it may be possible to measure some but not others
    depending on the data source. ADRIOs should avoid reporting DownloadActivity
    where all fields are None, and should instead just report None for
    `AdrioProgress.download`"""

    total: int | None
    """How many bytes are expected to be downloaded?"""
    downloaded: int | None
    """How many bytes have been downloaded?"""
    download_speed: int | None
    """What is the current approximate download speed?"""


class AdrioProgress(NamedTuple):
    """The payload of AdrioEvents.on_adrio_progress

    Perhaps not all ADRIOs will report progress, but those that do
    should emit one event when they start (with `ratio_complete` == 0)
    and one when they finish (with `ratio_complete` == 1). They are free
    to report as many intermediate progress events as they like."""

    adrio_name: str
    """The name of the ADRIO."""
    # attribute: AbsoluteName # TODO: see if we can get AbsoluteName here...
    # """The name of the attribute."""
    final: bool
    """Is this the last progress update for this ADRIO?"""
    ratio_complete: float
    """What ratio complete is it? (0: just started; 1: done)"""
    download: DownloadActivity | None
    """Download activity if any, and if it can be measured."""
    duration: float | None
    """If complete, how long did the ADRIO take overall (in seconds)?"""


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
    """
    Event fires at the start of the movement processing phase for every simulation tick.
    """

    on_movement_clause: Event[OnMovementClause]
    """Event fires after processing each active movement clause."""

    on_movement_finish: Event[OnMovementFinish]
    """
    Event fires at the end of the movement processing phase for every simulation tick.
    """

    # ADRIO Events
    on_adrio_progress: Event[AdrioProgress]
    """Event fires when an ADRIO is fetching data."""

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
        self.on_adrio_progress = Event()
