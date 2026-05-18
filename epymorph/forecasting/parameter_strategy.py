from abc import abstractmethod
from dataclasses import dataclass, field
from typing import OrderedDict, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ParameterStrategy:
    parameters: dict
    """A dictionary of unknown parameters."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of parameters."""
    grouping: None
    """Grouping for parameters (unimplemented)."""
    aggregation: None
    """Aggregation for parameters (unimplemented)."""

    def disambiguate(self) -> OrderedDict[str, str]:
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
        """Since grouping is not currently supported this
        is identical to disambiguate."""
        return self.disambiguate()


@dataclass(frozen=True)
class ParameterSelection(ParameterStrategy):
    parameters: dict
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
        return [str(k) for k, v in self.parameters]  # type: ignore


class ParameterSelector:
    """A list of estimated parameters."""

    def __init__(self, parameters):
        self.parameters = parameters

    def _mask(
        self,
        parameters_boolean: bool | list[bool] = False,
    ) -> NDArray[np.bool_]:
        m = np.zeros(shape=len(self.parameters.keys()), dtype=np.bool_)
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

    def by_name(self, *patterns: str) -> ParameterSelection:
        """
        Select parameters by name.

        Returns
        -------
        :
            The selection object.
        """
        if len(patterns) == 0:
            mask = self._mask(parameters_boolean=True)
        else:
            param_str = "*::*::{n}"
            mask = self._mask()
            for p in patterns:
                curr = self._mask(
                    parameters_boolean=[
                        (param_str.format(n=p) == str(name))
                        for name in self.parameters.keys()
                    ]
                )
                if not np.any(curr):
                    err = f"Pattern '{p}' did not match any parameters."
                    raise ValueError(err)
                mask |= curr

        return ParameterSelection(self.parameters, mask)
