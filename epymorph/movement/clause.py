from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick, TickDelta
from epymorph.context import Compartments


class ReturnClause:
    name = 'Return'

    def predicate(self, tick: Tick) -> bool:
        return True


RETURN = ReturnClause()

CompartmentPredicate = Callable[[list[str]], bool]


class TravelClause(ABC):
    name: str
    returns: TickDelta
    movement_mask: NDArray[np.bool_]

    @abstractmethod
    def predicate(self, tick: Tick) -> bool:
        pass


class ArrayClause(TravelClause):

    @abstractmethod
    def apply(self, tick: Tick) -> Compartments:
        pass


class RowClause(TravelClause):

    @abstractmethod
    def apply(self, tick: Tick, src_index: int) -> Compartments:
        pass


class CellClause(TravelClause):

    @abstractmethod
    def apply(self, tick: Tick, src_index: int, dst_index: int) -> int:
        pass


Clause = ReturnClause | ArrayClause | RowClause | CellClause
