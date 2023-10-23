"""
A static geo is one that is pre-packaged with all of its data; it doesn't need to fetch any data from outside itself,
and all of its data is resident in memory when loaded.
"""
from __future__ import annotations

import io
import os
import tarfile
from typing import cast

import jsonpickle
import numpy as np
from numpy.typing import NDArray

import epymorph.data_shape as shape
from epymorph.error import GeoValidationException
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, AttribDef, StaticGeoSpec,
                               validate_geo_values)
from epymorph.util import NDIndices


class StaticGeo(Geo[StaticGeoSpec]):
    """A Geo implementation which contains all of data pre-fetched and in-memory."""

    @staticmethod
    def load(tar_file: os.PathLike) -> StaticGeo:
        """Load a StaticGeo from tar format."""
        return StaticGeoFileOps.load_from_tar(tar_file)

    values: dict[str, NDArray]

    def __init__(self, spec: StaticGeoSpec, values: dict[str, NDArray]):
        self.values = values
        super().__init__(spec, len(values[LABEL.name]))

    def __getitem__(self, name: str) -> NDArray:
        if name not in self.values:
            raise KeyError(f"Attribute not found in geo: '{name}'")
        return self.values[name]

    @property
    def labels(self) -> NDArray[np.str_]:
        return self.values[LABEL.name]

    def validate(self) -> None:
        """
        Validate this geo against its specification.
        Raises GeoValidationException for any errors.
        """
        if self.spec.attribute_map.keys() != self.values.keys():
            raise GeoValidationException('Geo values do not match the given spec.')
        validate_geo_values(self.spec, self.values)

    def filter(self, selection: NDIndices) -> StaticGeo:
        """
        Create a new geo by selecting only certain nodes from another geo.
        Does not alter the original geo.
        """

        def select(attrib: AttribDef) -> NDArray:
            """Perform selections on attribute arrays."""
            arr = self.values[attrib.name]
            match attrib.shape:
                # it's possible not all of these shapes really make sense in a geo,
                # but not too painful to support them anyway
                case shape.Arbitrary(_):
                    return arr
                case shape.Node():
                    return arr[selection]
                case shape.NodeAndNode():
                    return arr[selection[:, np.newaxis], selection]
                case shape.Time():
                    return arr
                case shape.TimeAndNode():
                    return arr[:, selection]
                case shape.NodeAndArbitrary(_):
                    return arr[selection, :]
                case shape.TimeAndArbitrary(_):
                    return arr
                case shape.TimeAndNodeAndArbitrary(_):
                    return arr[:, selection, :]
                case x:
                    raise ValueError(f"Unsupported shape {x}")

        filtered_values = {
            attrib.name: select(attrib)
            for attrib in self.spec.attributes
        }
        return StaticGeo(self.spec, filtered_values)

    def save(self, file: os.PathLike) -> None:
        """Saves this geo to tar format."""
        StaticGeoFileOps.save_as_tar(self, file)


class StaticGeoFileOps:
    """Helper functions for saving and loading static geos as files."""

    @staticmethod
    def get_tar_filename(geo_id: str) -> str:
        """Returns the standard filename for a geo tar."""
        return f"{geo_id}.geo.tar"

    @staticmethod
    def save_as_tar(geo: StaticGeo, file: os.PathLike) -> None:
        """Save a StaticGeo to its tar format."""

        # Write the data file in memory
        npz_file = io.BytesIO()
        np.savez_compressed(npz_file, **geo.values)

        # Write the spec file in memory
        geo_file = io.BytesIO()
        geo_json = cast(str, jsonpickle.encode(geo.spec, unpicklable=True))
        geo_file.write(geo_json.encode('utf-8'))

        # Write the tar to disk
        with tarfile.open(file, 'w') as tar:
            def add_file(contents: io.BytesIO, name: str) -> None:
                info = tarfile.TarInfo(name)
                info.size = contents.tell()
                contents.seek(0)
                tar.addfile(info, contents)

            add_file(npz_file, 'data.npz')
            add_file(geo_file, 'spec.geo')

    @staticmethod
    def load_from_tar(file: os.PathLike) -> StaticGeo:
        """Load a StaticGeo from its tar format."""
        try:
            # Read the tar file into memory
            tar_buffer = io.BytesIO()
            with open(file, 'rb') as f:
                tar_buffer.write(f.read())
            tar_buffer.seek(0)

            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                npz_file = tar.extractfile(tar.getmember('data.npz'))
                geo_file = tar.extractfile(tar.getmember('spec.geo'))

                if npz_file is None or geo_file is None:
                    msg = 'Archive is missing data and/or spec files.'
                    raise GeoValidationException(msg)

                # Read the data file in memory
                with np.load(npz_file) as data:
                    values = dict(data)

                # Read the spec file in memory
                spec_json = geo_file.read().decode('utf8')
                spec = StaticGeoSpec.deserialize(spec_json)

                return StaticGeo(spec, values)

        except Exception as e:
            raise GeoValidationException(f"Unable to load '{file}' as a geo.") from e
