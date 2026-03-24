"""
This module extends the functionality of munge over an additional axis, the realizations.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, OrderedDict, Sequence

import numpy as np
from numpy.typing import NDArray

RealizationAggMethod = Literal["mean", "std", "quantiles"]
"""The supported methods for aggregating realizations."""


def quantile_method(q_val: float):
    def method(x):
        return x.quantile(q_val)

    method.__name__ = f"quantile_{round(100 * q_val, 1)}"
    return method


_every_025 = np.linspace(0.0, 1.0, 41)

"""Dictionaries of aggregation methods."""
agg_methods = {
    "mean": ["mean"],
    "std": ["std"],
    "median": ["median"],
    "quantiles": list(map(quantile_method, _every_025)),
}

"""Extend agg methods to individual quantiles."""
for q_val in _every_025:
    agg_methods[f"quantile_{round(100 * q_val, 1)}"] = [quantile_method(q_val)]


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
    """The number of realizations in the original output. """
    selection: NDArray[np.int_]
    """An integer array indicating which realization indices are selected. """
    aggregation: None = field(init=False, default=None)

    def agg(self, aggs) -> RealizationAggregation:
        if len(aggs) == 0:
            raise ValueError("No realization aggregation method supplied.")
        if any((agg := name) not in agg_methods for name in aggs):
            raise ValueError(f"{agg} is not a supported aggregator. ")
        agg_list = [m for name in aggs for m in agg_methods[name]]
        return RealizationAggregation(self.num_realizations, self.selection, agg_list)

    def mean(self) -> RealizationAggregation:
        return self.agg(["mean"])

    def std(self) -> RealizationAggregation:
        return self.agg(["std"])

    def quantiles(self) -> RealizationAggregation:
        return self.agg(["quantiles"])


class RealizationSelector:
    """
    A utility class for selecting a subset of the realizations. Most of the time you
    obtain one of these using `ParticleFilterOutput's` `select` property.
    """

    _num_realizations: int
    """The number of realizations in the original output. """

    def __init__(self, num_realizations: int):
        self._num_realizations = num_realizations

    def all(self) -> RealizationSelection:
        """
        Select all realizations.

        Returns
        -------
        :
            The selection object.
        """
        return RealizationSelection(
            self._num_realizations, np.arange(self._num_realizations)
        )

    def sample(
        self, *, rng: np.random.Generator, size: np.int_
    ) -> RealizationSelection:
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
            raise ValueError(
                (
                    "Sample size must be greater than zero "
                    "and less than the number of realizations. "
                )
            )

        if not isinstance(rng, np.random.Generator):
            raise ValueError("Parameter rng must be a NumPy random number generator. ")

        indices = rng.choice(np.arange(self._num_realizations), size, replace=False)

        return RealizationSelection(self._num_realizations, indices)


@dataclass(frozen=True)
class ParameterStrategy:
    param_names: list[str]
    """A list of estimated parameter names."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of parameters."""
    grouping: None
    """Grouping for parameters (unimplemented)."""
    aggregation: None
    """Aggregation for parameters (unimplemented)."""

    def disambiguate(self) -> OrderedDict[str, str]:
        selected = [(i, q) for i, q in enumerate(self.param_names) if self.selection[i]]
        qs_original = [str(q) for i, q in selected]
        qs_renamed = [f"{str(q)}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))

    @property
    def selected(self):
        """The quantities from the IPM which are selected, prior to any grouping."""
        return [q for sel, q in zip(self.selection, self.param_names) if sel]

    @property
    @abstractmethod
    def labels(self) -> Sequence[str]:
        """Labels for the parameters in the result, after any grouping."""

    def disambiguate_groups(self) -> OrderedDict[str, str]:
        return self.disambiguate()


@dataclass(frozen=True)
class ParameterSelection(ParameterStrategy):
    param_names: list
    """A list of estimated parameter names."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of parameters."""
    grouping: None = field(init=False, default=None)
    """Grouping for parameters (unimplemented)."""
    aggregation: None = field(init=False, default=None)
    """Aggregation for parameters (unimplemented)."""

    @property
    def labels(self) -> Sequence[str]:
        """Labels for the parameters in the result, after any grouping."""
        return self.selected


class ParameterSelector:
    _param_names: list[str]
    """A list of estimated parameter names."""

    def __init__(self, param_names: list[str]):
        self._param_names = param_names

    def _mask(
        self,
        parameters: bool | list[bool] = False,
    ) -> NDArray[np.bool_]:
        m = np.zeros(shape=len(self._param_names), dtype=np.bool_)
        if parameters is not False:
            m[:] = parameters

        return m

    def all(self) -> ParameterSelection:
        """
        Select all parameters.

        Returns
        -------
        :
            The selection object.
        """
        m = self._mask(parameters=True)
        return ParameterSelection(self._param_names, m)

    def by_name(self, *patterns: str) -> ParameterSelection:
        """
        Select parameters by name.

        Returns
        -------
        :
            The selection object.
        """
        if len(patterns) == 0:
            mask = self._mask(parameters=True)
        else:
            mask = self._mask()
            for p in patterns:
                curr = self._mask(
                    parameters=[(p == name) for name in self._param_names]
                )
                if not np.any(curr):
                    err = f"Pattern '{p}' did not match any parameters."
                    raise ValueError(err)
                mask |= curr

        return ParameterSelection(self._param_names, mask)
