"""
This module extends the functionality of munge over an additional axis, the realizations.
"""

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
from numpy.typing import NDArray

RealizationAggMethod = Literal["mean","std"]
"""The supported methods for aggregating realizations."""

@dataclass(frozen=True)
class RealizationStrategy:
    """
    A strategy for dealing with the realization axis, e.g., in processing results. 
    Strategies can include random sampling and aggregation. 
    
    Parameters
    ----------
    num_realizations: 
        The number of realizations. 
    selection: 
        A NDArray of integer indices to sample. 
    grouping: 
        Performs no operation, left in for consistency with other strategy classes. 
    aggregation: 
        A method or methods for aggregating the quantity groups. 
    """

    num_realizations: int
    """The number of realizations in the original output. """
    selection: NDArray[np.int_]
    """An integer array indicating which realization indices are selected. """
    aggregation: List[RealizationAggMethod] | None
    """A list of methods for aggregating the realization data."""

@dataclass(frozen=True)
class RealizationAggregation(RealizationStrategy):
    """
    A kind of `RealizationStrategy` describing an aggregate operation on realizations.

    Typically you will create one of these by calling methods on a `RealizationSelector`
    instance.

    Parameters
    ----------
    num_realizations :
        The number of particle realizations.
    selection :
        A boolean mask for selection of a subset of realizations.
    aggregation :
        A method for aggregating the realizations.
    """

    num_realizations: int
    """The number of realizations in the original output. """
    selection: NDArray[np.int_]
    """An integer array indicating which realization indices are selected. """
    aggregation: List[RealizationAggMethod]
    """A list of methods for aggregating the realization data."""


@dataclass(frozen=True)
class RealizationSelection(RealizationStrategy):

    """
    A kind of `RealizationStrategy` describing a sub-selection of realizations.
    A selection performs no grouping or aggregation. 
    """

    num_realizations: int
    selection: NDArray[np.int_]
    aggregation: None = field(init = False, default = None)

    def agg(self,*aggs: RealizationAggMethod)->RealizationAggregation:
        if len(aggs) == 0:
            raise ValueError("No realization aggregation method supplied.")

        return RealizationAggregation(self.num_realizations,self.selection,[*aggs])

    def mean(self) -> RealizationAggregation:
        return self.agg("mean")

    def std(self)->RealizationAggregation:
        return self.agg("std")


class RealizationSelector:
    """
    A utility class for selecting a subset of the realizations. Most of the time you 
    obtain one of these using `ParticleFilterOutput's` `select` property. 
    """

    _num_realizations: int
    """The number of realizations in the original output. """

    def __init__(self,num_realizations: int):
        self._num_realizations = num_realizations

    def all(self)->RealizationSelection:
        """
        Select all realizations. 

        Returns
        -------
        : 
            The selection object. 
        """
        return RealizationSelection(self._num_realizations,
                                     np.arange(self._num_realizations))

    def sample(self,*,rng:np.random.Generator,size:np.int_)->RealizationSelection:
        """
        Select a random subset of the realizations. 

        Parameters
        ----------
        rng : 
            The numpy random generator to sample from. 
        size : 
            The number of samples to draw. 

        Returns
        -------
        : 
            The selection object. 
        """
        if (size <= 0) or (size > self._num_realizations):
            raise ValueError(("Sample size must be greater ",
            "than zero and less than the number of realizations. "))

        return RealizationSelection(self._num_realizations,
                                     rng.choice(np.arange(self._num_realizations),size))
