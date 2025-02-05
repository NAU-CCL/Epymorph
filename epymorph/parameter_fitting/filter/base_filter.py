"""
This module defines the abstract base class for filters used in the application.

The `BaseFilter` class provides a template for creating various filter implementations
with customizable parameters. Subclasses must implement the `run` method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from epymorph.parameter_fitting.output import ParticleFilterOutput


class BaseFilter(ABC):
    """
    Abstract base class for filters.

    Attributes
    ----------
    params : Dict[str, Any]
        Parameters for the filter.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the filter with the given parameters.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments representing filter parameters.
        """
        self.params: Dict[str, Any] = kwargs

    @abstractmethod
    def run(self, *args, **kwargs: Any) -> ParticleFilterOutput:
        """
        Abstract method to run the filter. Must be implemented by subclasses.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments needed to run the filter.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")
