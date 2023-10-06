"""
A static geo is one that is pre-packaged with all of its data; it doesn't need to fetch any data from outside itself,
and all of its data is resident in memory when loaded.
"""
from __future__ import annotations

from os import PathLike
from types import MappingProxyType
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from epymorph.geo.common import AttribDef
from epymorph.geo.geo import LABEL, Geo
from epymorph.util import NDIndices, shape_matches


class StaticGeo(Geo):
    """A Geo implementation which contains all of data pre-fetched and in-memory."""

    @classmethod
    def load(cls, npz_file: PathLike) -> StaticGeo:
        """Load a StaticGeo from its .npz format."""
        with np.load(npz_file) as npz_data:
            values = dict(npz_data)
            return StaticGeo.from_values(values)

    @classmethod
    def from_values(cls, values: dict[str, NDArray]) -> StaticGeo:
        """Create a Geo containing the given values as attributes."""
        return cls(
            # Infer AttribDefs from the given values
            # TODO: this breaks for structural types (like CentroidDType). `.type` comes back as `np.void` which isn't ideal
            attrib_defs=[AttribDef(name, v.dtype.type)
                         for name, v in values.items()],
            attrib_values=values
        )

    _values: dict[str, NDArray]

    def __init__(self, attrib_defs: Iterable[AttribDef], attrib_values: dict[str, NDArray]):
        attributes = MappingProxyType({a.name: a for a in attrib_defs})
        Geo.validate_attributes(attributes.values())
        Geo.validate_values(Geo.required_attributes, attrib_values)

        self._values = {
            name: values
            for name, values in attrib_values.items()
            if name in attributes  # weed out extra values
        }

        nodes = len(attrib_values[LABEL.name])
        super().__init__(attributes, nodes)

    def __getitem__(self, name: str) -> NDArray:
        if name not in self._values:
            raise KeyError(f"Attribute not found in geo: '{name}'")
        return self._values[name]

    def filter(self, selection: NDIndices) -> StaticGeo:
        """
        Create a new geo by selecting only certain nodes from another geo.
        Does not alter the original geo.
        """

        n = self.nodes

        def select(arr: NDArray) -> NDArray:
            # Handle selections on attribute arrays
            if shape_matches(arr, (n,)):
                return arr[selection]
            elif shape_matches(arr, (n, n)):
                return arr[selection[:, np.newaxis], selection]
            elif shape_matches(arr, ('?', n)):  # matches TxN
                return arr[:, selection]
            elif shape_matches(arr, (n, '?')):  # matches NxA
                return arr[selection, :]
            else:
                raise Exception(f"Unsupported shape {arr.shape}.")

        filtered_values = {
            attrib_name: select(self._values[attrib_name])
            for attrib_name, attrib in self.attributes.items()
        }
        return StaticGeo(self.attributes.values(), filtered_values)

    def save(self, npz_file: PathLike) -> None:
        """Saves a StaticGeo to .npz format."""
        np.savez_compressed(npz_file, **self._values)
