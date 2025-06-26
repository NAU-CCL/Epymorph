"""
Capturing extremely detailed movement data from a simulation, which can be useful for
debugging movement models.
"""

from abc import abstractmethod
from contextlib import contextmanager
from typing import Generator, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_type import SimDType
from epymorph.event import EventBus, OnMovementClause, OnStart
from epymorph.rume import RUME
from epymorph.util import subscriptions

_events = EventBus()


class MovementData(Protocol):
    """
    Container for collected simulation movement data. Run the simulation inside a
    `movement_data` context and an instance of this class will be returned. After
    the simulation has completed, you can use this object to retrieve movement data
    either by clause or in aggregate across all clauses.

    A note about axis ordering: both `requested` and `actual` data are NxN in shape,
    where N is the number of geo nodes in the simulation. We use the convention that
    the first axis represents where the movement is "from" and the second axis
    represents where the movement is "to". Movement due to the return clause is treated
    the same way: returning from work to home for example is recorded as (work, home).
    """

    @abstractmethod
    def requested_by(self, clause: str) -> NDArray[SimDType]:
        """
        Retrieve the time series (steps) of requested movement by clause.

        Parameters
        ----------
        clause :
            The clause to fetch.

        Returns
        -------
        :
            The movement data as a (S,N,N)-shaped array.
        """

    @abstractmethod
    def actual_by(self, clause: str) -> NDArray[SimDType]:
        """
        Retrieve the time series (steps) of actual movement by clause.

        Parameters
        ----------
        clause :
            The clause to fetch.

        Returns
        -------
        :
            The movement data as a (S,N,N,C)-shaped array.
        """

    @abstractmethod
    def requested_all(self) -> NDArray[SimDType]:
        """
        Retrieve the time series (steps) of requested movement for all clauses.

        Returns
        -------
        :
            The movement data as a (S,N,N)-shaped array.
        """

    @abstractmethod
    def actual_all(self) -> NDArray[SimDType]:
        """
        Retrieve the time series (steps) of actual movement for all clauses.

        Returns
        -------
        :
            The movement data as a (S,N,N,C)-shaped array.
        """


class _Entry(NamedTuple):
    """
    A generic container for any sort of movement data associated with a
    particular clause firing on a particular tick.
    """

    name: str
    """The clause name."""
    tick: int
    """The tick index."""
    data: NDArray[SimDType]
    """The movement data."""


class _MovementDataBuilder(MovementData):
    """The internal implementation of the `MovementData` protocol."""

    # Implementation note:
    #
    # Context managers make returning values at the end pretty awkward, while returning
    # a value right away is easy. But doing that here means the end user gets access to
    # the data container before it's finished, which could cause confusion if the user
    # tries to access it while the simulation is in progress.
    #
    # To mitigate that, this implementation keeps a `ready` flag and uses that to
    # prevent premature access to the data.

    _rume: RUME | None = None
    _requested = list[_Entry]()
    _actual = list[_Entry]()
    _ready: bool = False

    def _set_rume(self, rume: RUME) -> None:
        if self._rume is not None:
            err = (
                "Invalid state: `movement_data` should only be used to capture one "
                "simulation run."
            )
            raise RuntimeError(err)
        self._rume = rume

    def _record(self, clause_name: str, tick: int, requested, actual) -> None:
        # Don't allow recording after "ready".
        if self._ready:
            err = (
                "Invalid state: `movement_data` should only be used to capture one "
                "simulation run."
            )
            raise RuntimeError(err)
        self._requested.append(_Entry(clause_name, tick, requested))
        self._actual.append(_Entry(clause_name, tick, actual))

    def _get_dim_if_ready(self) -> tuple[int, int, int]:
        # Returns the lengths of axes (S,N,C)
        if not self._ready or self._rume is None:
            err = (
                "Invalid state: `movement_data` results cannot be accessed until the "
                "simulation is complete."
            )
            raise RuntimeError(err)
        rume = self._rume
        return (rume.num_ticks, rume.scope.nodes, rume.ipm.num_compartments)

    @override
    def requested_by(self, clause: str) -> NDArray[SimDType]:
        S, N, _ = self._get_dim_if_ready()
        result = np.zeros((S, N, N), dtype=SimDType)
        for name, tick, data in self._requested:
            if name == clause:
                result[tick, :, :] = data
        return result

    @override
    def actual_by(self, clause: str) -> NDArray[SimDType]:
        S, N, C = self._get_dim_if_ready()
        result = np.zeros((S, N, N, C), dtype=SimDType)
        for name, tick, data in self._actual:
            if name == clause:
                result[tick, :, :, :] = data
        return result

    @override
    def requested_all(self) -> NDArray[SimDType]:
        S, N, _ = self._get_dim_if_ready()
        result = np.zeros((S, N, N), dtype=SimDType)
        for _name, tick, data in self._requested:
            result[tick, :, :] += data
        return result

    @override
    def actual_all(self) -> NDArray[SimDType]:
        S, N, C = self._get_dim_if_ready()
        result = np.zeros((S, N, N, C), dtype=SimDType)
        for _name, tick, data in self._actual:
            result[tick, :, :, :] += data
        return result


@contextmanager
def movement_data() -> Generator[MovementData, None, None]:
    """
    Run one simulation in this context manager in order to capture
    detailed movement data. This returns a `MovementData` object which
    can be used -- after the context exits -- to retrieve the movement data.

    Yields
    ------
    :
        The object that will contain the collected movement data.

    Examples
    --------
    ```python
    with movement_data() as my_data:
        # don't access `my_data` yet, it hasn't recorded anything!
        sim = BasicSimulator(rume)
        my_results = sim.run()

    # now you can use `my_data`
    total_movement = my_data.actual_all()
    ```
    """
    md = _MovementDataBuilder()

    def on_start(e: OnStart):
        nonlocal md
        md._set_rume(e.rume)

    def on_clause(e: OnMovementClause):
        nonlocal md
        md._record(e.clause_name, e.tick, e.requested, e.actual)

    with subscriptions() as sub:
        sub.subscribe(_events.on_start, on_start)
        sub.subscribe(_events.on_movement_clause, on_clause)
        yield md
        # Once the context completes, the simulation should be done.
        md._ready = True
