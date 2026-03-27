from dataclasses import dataclass

from numpy.typing import NDArray

from epymorph.forecasting.dynamic_params import ParamFunctionDynamics, Prior


@dataclass(frozen=True)
class UnknownParam:
    """
    Contains the information for an unknown parameter. An unknown parameter is a
    parameter which can vary across realizations in a multi-realization simulation. Some
    simulators will try to estimate unknown parameters.
    """

    prior: NDArray | Prior
    """
    The prior distribution or initial values of the parameter.
    """

    dynamics: ParamFunctionDynamics
    """
    The dynamics of the parameter dictating how the parameter changes over time.
    """
