"""
A fully-configured RUME has a context that can be used to interact
with simulation data, for example, accessing geo and parameter attributes,
calculating the simulation clock, initializing the world state, and so on.
"""
import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import CompartmentModel
from epymorph.data_type import AttributeArray
from epymorph.geo.geo import Geo
from epymorph.params import NormalizedParams, NormalizedParamsDict
from epymorph.simulation import (CachingGetAttributeMixin, GeoData,
                                 SimDimensions)


class RumeContext(CachingGetAttributeMixin):
    """The fully-realized configuration and data we need to run a simulation."""
    dim: SimDimensions
    rng: np.random.Generator
    version: int
    """
    `version` indicates when changes have been made to the context.
    If `version` hasn't changed, no other changes have been made.
    """
    _geo: Geo
    _ipm: CompartmentModel
    _params: NormalizedParamsDict

    def __init__(
        self,
        dim: SimDimensions,
        rng: np.random.Generator,
        geo: Geo,
        ipm: CompartmentModel,
        params: NormalizedParamsDict,
    ):
        self.dim = dim
        self.rng = rng
        self.version = 0
        self._geo = geo
        self._ipm = ipm
        self._params = params
        CachingGetAttributeMixin.__init__(self, geo, params, dim)

    def update_param(self, attr_name: str, value: AttributeArray) -> None:
        """Updates a params value."""
        self._params[attr_name] = value.copy()
        self.clear_attribute_getter(attr_name)
        self.version += 1

    @property
    def geo(self) -> GeoData:
        """The Geo."""
        return self._geo

    @property
    def compartment_mobility(self) -> NDArray[np.bool_]:
        """Which compartments from the IPM are subject to movement?"""
        return np.array(
            ['immobile' not in c.tags for c in self._ipm.compartments],
            dtype=np.bool_
        )

    @property
    def params(self) -> NormalizedParams:
        """The params values."""
        return self._params
