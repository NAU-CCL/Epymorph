"""
A static geo is one that is pre-packaged with all of its data; it doesn't need to fetch any data from outside itself,
and all of its data is resident in memory when loaded.
"""
from importlib.abc import Traversable
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Iterator, Self, cast

import numpy as np
from jsonpickle import encode as json_encode
from numpy.typing import NDArray

import epymorph.data_shape as shape
from epymorph.cache import load_bundle, save_bundle
from epymorph.error import AttributeException, GeoValidationException
from epymorph.geo.geo import Geo
from epymorph.geo.spec import LABEL, StaticGeoSpec, validate_geo_values
from epymorph.simulation import AttributeArray, AttributeDef
from epymorph.util import NDIndices, as_sorted_dict

_STATIC_GEO_CACHE_VERSION = 2


class StaticGeo(Geo[StaticGeoSpec]):
    """A Geo implementation which contains all of data pre-fetched and in-memory."""

    values: dict[str, AttributeArray]

    def __init__(self, spec: StaticGeoSpec, values: dict[str, NDArray]):
        if not LABEL.name in values or not np.issubdtype(values[LABEL.name].dtype, np.str_):
            msg = "Geo must contain an attribute called 'label' of type string."
            raise ValueError(msg)
        self.values = values
        super().__init__(spec, len(values[LABEL.name]))

    def __getitem__(self, name: str, /) -> AttributeArray:
        if name not in self.values:
            raise AttributeException(f"Attribute not found in geo: '{name}'")
        return self.values[name]

    def __contains__(self, name: str, /) -> bool:
        return name in self.values

    @property
    def labels(self) -> NDArray[np.str_]:
        """The labels for every node in this geo."""
        return self.values[LABEL.name]  # type: ignore (constructor check should be sufficient)

    def validate(self) -> None:
        """
        Validate this geo against its specification.
        Raises GeoValidationException for any errors.
        """
        if self.spec.attribute_map.keys() != self.values.keys():
            raise GeoValidationException('Geo values do not match the given spec.')
        validate_geo_values(self.spec, self.values)

    def filter(self, selection: NDIndices) -> Self:
        """
        Create a new geo by selecting only certain nodes from another geo.
        Does not alter the original geo.
        """

        def select(attrib: AttributeDef) -> NDArray:
            """Perform selections on attribute arrays."""
            arr = self.values[attrib.name]
            match attrib.shape:
                # it's possible not all of these shapes really make sense in a geo,
                # but not too painful to support them anyway
                case shape.Node():
                    return arr[selection]
                case shape.NodeAndNode():
                    return arr[selection[:, np.newaxis], selection]
                case shape.NodeAndCompartment():
                    return arr[selection, :]
                case shape.Time():
                    return arr
                case shape.TimeAndNode():
                    return arr[:, selection]
                case x:
                    raise ValueError(f"Unsupported shape {x}")

        filtered_values = {
            attrib.name: select(attrib)
            for attrib in self.spec.attributes
        }
        return self.__class__(self.spec, filtered_values)

    def save(self, file: PathLike) -> None:
        """Saves this geo to tar format."""
        StaticGeoFileOps.save_as_archive(self, file)


class StaticGeoFileOps:
    """Helper functions for saving and loading static geos as files."""

    @staticmethod
    def to_archive_filename(geo_id: str) -> str:
        """Returns the standard filename for a geo archive."""
        return f"{geo_id}.geo.tgz"

    @staticmethod
    def to_geo_name(filename: str) -> str:
        """Returns the geo ID from a standard geo archive filename."""
        return filename.removesuffix('.geo.tgz')

    @staticmethod
    def iterate_dir(directory: Traversable) -> Iterator[tuple[Traversable, str]]:
        """
        Iterates through the given directory non-recursively, returning all archived geos.
        Each item in the returned iterator is a tuple containing:
        1. the Traversable instance for the file itself, and 
        2. the geo's ID.
        """
        return ((f, StaticGeoFileOps.to_geo_name(f.name))
                for f in directory.iterdir()
                if f.is_file() and f.name.endswith('.geo.tgz'))

    @staticmethod
    def iterate_dir_path(directory: Path) -> Iterator[tuple[Path, str]]:
        """
        Iterates through the given directory non-recursively, returning all archived geos.
        Each item in the returned iterator is a tuple containing:
        1. the Path for the file itself, and 
        2. the geo's ID.
        """
        return ((f, StaticGeoFileOps.to_geo_name(f.name))
                for f in directory.iterdir()
                if f.is_file() and f.name.endswith('.geo.tgz'))

    @staticmethod
    def save_as_archive(geo: StaticGeo, file: PathLike) -> None:
        """Save a StaticGeo to its tar format."""

        # Write the data file
        # (sorting the geo values makes the sha256 a little more stable)
        npz_file = BytesIO()
        np.savez_compressed(npz_file, **as_sorted_dict(geo.values))

        # Write the spec file
        geo_file = BytesIO()
        geo_json = cast(str, json_encode(geo.spec, unpicklable=True))
        geo_file.write(geo_json.encode('utf-8'))

        save_bundle(
            to_path=file,
            version=_STATIC_GEO_CACHE_VERSION,
            files={
                "data.npz": npz_file,
                "spec.geo": geo_file,
            },
        )

    @staticmethod
    def load_from_archive(file: PathLike) -> StaticGeo:
        """Load a StaticGeo from its tar format."""
        try:
            files = load_bundle(file, version_at_least=_STATIC_GEO_CACHE_VERSION)
            if "data.npz" not in files or "spec.geo" not in files:
                msg = 'Archive is incomplete: missing data, spec, and/or checksum files.'
                raise GeoValidationException(msg)

            # Read the spec file
            geo_file = files["spec.geo"]
            geo_file.seek(0)
            spec_json = geo_file.read().decode('utf8')
            spec = StaticGeoSpec.deserialize(spec_json)

            # Read the data file
            npz_file = files["data.npz"]
            npz_file.seek(0)
            with np.load(npz_file) as data:
                values = dict(data)

            return StaticGeo(spec, values)
        except Exception as e:
            raise GeoValidationException(f"Unable to load '{file}' as a geo.") from e
