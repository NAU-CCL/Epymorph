"""
epymorph's event framework.

epymorph components may define a set of events relevant to their function,
and during processing can publish to the singleton `EventBus` for subscription
and processing by other systems.
"""

from typing import NamedTuple

from numpy.typing import NDArray

from epymorph.attribute import AbsoluteName
from epymorph.data_type import SimDType
from epymorph.rume import RUME
from epymorph.util import Event, Singleton

#####################
# Simulation Events #
#####################


class OnStart(NamedTuple):
    """Event payload for the start of a simulation."""

    simulator: str
    """Name of the simulator class."""
    rume: RUME
    """The RUME for the simulation."""


class OnTick(NamedTuple):
    """Event payload for the completion of a simulation tick."""

    tick_index: int
    """The index of the just-completed tick."""
    ticks: int
    """The total number of ticks for the simulation."""


###################
# Movement Events #
###################


class OnMovementStart(NamedTuple):
    """Event payload for the start of movement processing during a tick."""

    tick: int
    """Which simulation tick."""
    day: int
    """Which simulation day."""
    step: int
    """Which tau step (by index)."""


class OnMovementClause(NamedTuple):
    """Event payload for the completion of processing a movement clause."""

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
    """Event payload for the completion of movement processing during a tick."""

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
    """
    Part of the `ADRIOProgress` event payload describing ADRIO network activity.

    All fields are optional, as it may be possible to measure some but not others
    depending on the data source. ADRIOs should avoid reporting empty activity
    events and instead instead just report `None` (see `ADRIO` methods).
    """

    total: int | None
    """How many bytes are expected to be downloaded?"""
    downloaded: int | None
    """How many bytes have been downloaded?"""
    download_speed: int | None
    """What is the current approximate download speed?"""


class ADRIOProgress(NamedTuple):
    """
    Event payload for ADRIO data processing progress.

    Not all ADRIOs will be able to report progress meaningfully, but those that do
    should emit one event when they start (with `ratio_complete` == 0)
    and one when they finish (with `ratio_complete` == 1). They are free
    to report as many intermediate progress events as they like.
    """

    adrio_name: str
    """The full name of the ADRIO class."""
    attribute: AbsoluteName
    """The name of the attribute being evaluated."""
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
    """
    A singleton for epymorph event publishing and subscription.

    Obtain a reference to the global event bus by constructing it as usual --
    `EventBus()` -- but be aware that this will produce the same instance every time.
    """

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
    on_adrio_progress: Event[ADRIOProgress]
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
