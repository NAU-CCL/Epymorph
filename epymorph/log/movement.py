"""For capturing extremely detailed movement data from a simulation."""

from abc import abstractmethod
from contextlib import contextmanager
from typing import Generator, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.event import EventBus, OnMovementClause, OnStart
from epymorph.rume import RUME
from epymorph.util import subscriptions

_events = EventBus()


class MovementData(Protocol):
    """
    A data collection of simulation movement data.
    Run the simulation inside a `movement_data` context and
    an instance of this class will be returned. After the simulation
    has completed, you can use this object to retrieve movement data
    either by clause or in aggregate across all clauses.

    A note about axis ordering: both `requested` and `actual` data
    includes a pair of axes that are the length of the number of geo nodes
    in the simulation (N). Because these data represent movement flows we
    use the convention that the first N represents where the movement is "from" and
    the second N represents where the movement is "to". This is true regardless
    of which clause is responsible. Returning movement from the return clause is treated
    no differently than outgoing movement from a user-defined clause.
    """

    @abstractmethod
    def requested_by(self, clause: str) -> NDArray[SimDType]:
        """The time series of requested movement by clause. Array shape: (T,N,N)"""

    @abstractmethod
    def actual_by(self, clause: str) -> NDArray[SimDType]:
        """The time series of actual movement by clause. Array shape: (T,N,N,C)"""

    @abstractmethod
    def requested_all(self) -> NDArray[SimDType]:
        """
        The time series of requested movement for all clauses. Array shape: (T,N,N)
        """

    @abstractmethod
    def actual_all(self) -> NDArray[SimDType]:
        """The time series of actual movement for all clauses. Array shape: (T,N,N,C)"""


class _Entry(NamedTuple):
    """The data associated with a movement clause firing on a given tick."""

    name: str
    tick: int
    data: NDArray[SimDType]


class _MovementDataBuilder(MovementData):
    """
    The mechanics of a context require that, in order to return a value, we have to
    supply that value at the time we yield to the context body. Therefore we need
    this builder object so we can provide a stand-in that will be populated as the
    context body runs. We keep a `ready` flag to prevent access to the data before
    the simulation (and the context) has completed.
    """

    ready: bool
    rume: RUME | None
    requested: list[_Entry]
    actual: list[_Entry]

    def __init__(self):
        self.ready = False
        self.requested = []
        self.actual = []

    def _get_dim(self) -> tuple[int, int, int]:
        """
        Checks that the class is in a valid state and, if so, returns dimensions info.
        """
        if not self.ready or self.rume is None:
            msg = (
                "Invalid state: MovementData cannot be accessed until the simulation "
                "is complete."
            )
            raise RuntimeError(msg)

        return (
            self.rume.num_ticks,
            self.rume.scope.nodes,
            self.rume.ipm.num_compartments,
        )

    def requested_by(self, clause: str) -> NDArray[SimDType]:
        S, N, _ = self._get_dim()
        result = np.zeros((S, N, N), dtype=SimDType)
        for name, tick, data in self.requested:
            if name == clause:
                result[tick, :, :] = data
        return result

    def actual_by(self, clause: str) -> NDArray[SimDType]:
        S, N, C = self._get_dim()
        result = np.zeros((S, N, N, C), dtype=SimDType)
        for name, tick, data in self.actual:
            if name == clause:
                result[tick, :, :, :] = data
        return result

    def requested_all(self) -> NDArray[SimDType]:
        S, N, _ = self._get_dim()
        result = np.zeros((S, N, N), dtype=SimDType)
        for _name, tick, data in self.requested:
            result[tick, :, :] += data
        return result

    def actual_all(self) -> NDArray[SimDType]:
        S, N, C = self._get_dim()
        result = np.zeros((S, N, N, C), dtype=SimDType)
        for _name, tick, data in self.actual:
            result[tick, :, :, :] += data
        return result


@contextmanager
def movement_data() -> Generator[MovementData, None, None]:
    """
    Run a simulation in this context in order to collect detailed movement data
    throughout the simulation run. This returns a MovementData object which
    can be used -- after this context is exits -- to retrieve the movement data.
    """
    md = _MovementDataBuilder()

    def on_start(e: OnStart):
        nonlocal md
        md.rume = e.rume

    def on_clause(e: OnMovementClause):
        nonlocal md
        md.requested.append(_Entry(e.clause_name, e.tick, e.requested))
        md.actual.append(_Entry(e.clause_name, e.tick, e.actual))

    with subscriptions() as sub:
        sub.subscribe(_events.on_start, on_start)
        sub.subscribe(_events.on_movement_clause, on_clause)
        yield md
        md.ready = True
