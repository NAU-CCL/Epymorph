"""
This module extends the functionality of munge over an additional axis, realizations.
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, OrderedDict, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from epymorph.attribute import NamePattern

if TYPE_CHECKING:
    from epymorph.forecasting.pipeline import UnknownParam


def quantile_method(q_val: float) -> Callable[[pd.Series], np.floating]:
    """Returns a function that evaluates the given quantile."""

    def method(x: pd.Series) -> np.floating:
        return x.quantile(q_val)

    method.__name__ = f"quantile_{round(100 * q_val, 1)}"
    return method


_every_025 = np.linspace(0.0, 1.0, 41)

agg_methods: dict[str, list[Any]] = {
    "mean": ["mean"],
    "std": ["std"],
    "median": ["median"],
    "quantiles": [quantile_method(q_val) for q_val in _every_025],
    # Supported quantiles are every 2.5%
    **{
        f"quantile_{round(100 * q_val, 1)}": [quantile_method(q_val)]
        for q_val in _every_025
    },
}
"""Dictionaries of aggregation methods."""


@dataclass(frozen=True)
class ParameterStrategy:
    """
    A strategy for selecting estimated parameters from the output of a filtering run.

    Parameters
    ----------
    parameters:
        A mapping of NamePatterns to UnknownParam objects.
    selection:
        A boolean mask for parameter selection.
    """

    parameters: Mapping[NamePattern, "UnknownParam"]
    """A dictionary of unknown parameters."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of parameters."""
    grouping: None
    """Grouping for parameters (unimplemented)."""
    aggregation: None
    """Aggregation for parameters (unimplemented)."""

    def disambiguate(self) -> OrderedDict[str, str]:
        """
        Disambiguates parameter names to avoid any potential duplicates.
        Returns a dictionary from new names to original names.
        """
        selected = [
            (i, q) for i, q in enumerate(self.parameters.keys()) if self.selection[i]
        ]
        qs_original = [str(q) for i, q in selected]
        qs_renamed = [f"{str(q)}_{i}" for i, q in selected]
        return OrderedDict(zip(qs_renamed, qs_original))

    @property
    def selected(self) -> Sequence:
        """The parameters from the model which are selected, prior to any grouping."""
        return [q for sel, q in zip(self.selection, self.parameters.values()) if sel]

    @property
    @abstractmethod
    def labels(self) -> Sequence[str]:
        """Labels for the parameters in the result, after any grouping."""

    def disambiguate_groups(self) -> OrderedDict[str, str]:
        """
        Since grouping is not currently supported this is identical to disambiguate.
        """
        return self.disambiguate()


@dataclass(frozen=True)
class ParameterSelection(ParameterStrategy):
    parameters: Mapping[NamePattern, "UnknownParam"]
    """A dictionary of unknown parameters."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of parameters."""
    grouping: None = field(init=False, default=None)
    """Grouping for parameters (unimplemented)."""
    aggregation: None = field(init=False, default=None)
    """Aggregation for parameters (unimplemented)."""

    @property
    def labels(self) -> Sequence[str]:
        """Labels for the parameters in the result, after any grouping."""
        return [str(k) for k, v in self.parameters.items()]


class ParameterSelector:
    """
    A utility class for selecting a subset of the estimated parameters. Most of the time
    you obtain one of these using `FilterOutput`'s `param_select` property.
    """

    parameters: Mapping[NamePattern, "UnknownParam"]
    """A dictionary of unknown parameters."""

    def __init__(self, parameters: Mapping[NamePattern, "UnknownParam"]):
        self.parameters = parameters

    def _mask(
        self,
        parameters_boolean: bool | list[bool] = False,
    ) -> NDArray[np.bool_]:
        m = np.zeros(shape=len(self.parameters), dtype=np.bool_)
        if parameters_boolean is not False:
            m[:] = parameters_boolean

        return m

    def all(self) -> ParameterSelection:
        """
        Select all parameters.

        Returns
        -------
        :
            The selection object.
        """
        m = self._mask(parameters_boolean=True)
        return ParameterSelection(self.parameters, m)

    def by_name(self, *patterns: str | NamePattern) -> ParameterSelection:
        """
        Select parameters by name.

        Parameters
        ----------
        patterns :
            A list of parameter name patterns to match, as a var-arg.

        Returns
        -------
        :
            The selection object.
        """
        if len(patterns) == 0:
            mask = self._mask(parameters_boolean=True)
        else:
            mask = self._mask()
            for p in patterns:
                p = NamePattern.of(p)
                curr = self._mask(
                    parameters_boolean=[
                        (p.match(name)) for name in self.parameters.keys()
                    ]
                )
                if not np.any(curr):
                    err = f"Pattern '{str(p)}' did not match any parameters."
                    raise ValueError(err)
                mask |= curr

        return ParameterSelection(self.parameters, mask)


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
        An integer array indicating which realizations are selected.
    aggregation:
        A list of methods for aggregating the realization data.
    """

    num_realizations: int
    """The number of realizations in the original output. """
    selection: NDArray[np.intp]
    """An integer array indicating which realization indices are selected. """
    aggregation: List[str] | None
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
        The number of realizations.
    selection :
        An integer array indicating which realizations are selected.
    aggregation :
        A list of methods for aggregating the realization data.
    """

    num_realizations: int
    """The number of realizations in the original output."""
    selection: NDArray[np.intp]
    """An integer array indicating which realizations are selected."""
    aggregation: List[str]
    """A list of methods for aggregating the realization data."""


@dataclass(frozen=True)
class RealizationSelection(RealizationStrategy):
    """
    A kind of `RealizationStrategy` describing a sub-selection of realizations.
    A selection performs no grouping or aggregation.

    Parameters
    ----------
    num_realizations :
        The number of realizations.
    selection :
        An integer array indicating which realizations are selected.
    aggregation :
        A list of methods for aggregating the realization data.
    """

    num_realizations: int
    """The number of realizations in the original output."""
    selection: NDArray[np.intp]
    """An integer array indicating which realization indices are selected."""
    aggregation: None = field(init=False, default=None)
    """A list of methods for aggregating the realization data."""

    def agg(self, aggs: list[str]) -> RealizationAggregation:
        """
        Aggregate the selected realizations using the specified aggregation methods.

        Parameters
        ----------
        aggs :
            A list of aggregation method names. In general, one data series will be
            produced for each aggregation method. Supported methods are: "mean", "std",
            "median", "quantiles" (for a default set of quantiles), and a specified
            quantile in the format "quantile_<value>" where values are range from 0 to 1
            with a step of 2.5%, like "quantile_0.0", "quantile_2.5", etc.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        if len(aggs) == 0:
            raise ValueError("No realization aggregation method supplied.")
        if any((agg := name) not in agg_methods for name in aggs):
            raise ValueError(f"{agg} is not a supported aggregator. ")
        agg_list = [m for name in aggs for m in agg_methods[name]]
        return RealizationAggregation(self.num_realizations, self.selection, agg_list)

    def mean(self) -> RealizationAggregation:
        """
        Aggregate the selected realizations using the mean. Equivalent to
        `agg(["mean"])`.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return self.agg(["mean"])

    def std(self) -> RealizationAggregation:
        """
        Aggregate the selected realizations using the standard deviation. Equivalent to
        `agg(["std"])`.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return self.agg(["std"])

    def quantiles(self) -> RealizationAggregation:
        """
        Aggregate the selected realizations using a standard set of quantiles.
        Equivalent to `agg(["quantiles"])`.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return self.agg(["quantiles"])


class RealizationSelector:
    """
    A utility class for selecting a subset of the realizations. Most of the time you
    obtain one of these using `FilterOutput`'s `select` property.
    """

    _num_realizations: int
    """The number of realizations in the original output."""

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
            self._num_realizations,
            np.arange(self._num_realizations),
        )

    def sample(
        self,
        *,
        rng: np.random.Generator,
        size: int | np.integer,
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
                    "and less than the number of realizations."
                )
            )

        if not isinstance(rng, np.random.Generator):
            raise ValueError("Parameter rng must be a NumPy random number generator.")

        indices = rng.choice(self._num_realizations, size, replace=False)

        return RealizationSelection(self._num_realizations, indices)
