"""
A static geo is one that is pre-packaged with all of its data; it doesn't need to fetch any data from outside itself,
and all of its data is resident in memory when loaded.
"""
import hashlib
import io
import os
import tarfile
from importlib.abc import Traversable
from pathlib import Path
from typing import Iterator, Self, cast

import jsonpickle
import numpy as np
from numpy.typing import NDArray

import epymorph.data_shape as shape
from epymorph.error import AttributeException, GeoValidationException
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, AttribDef, StaticGeoSpec,
                               validate_geo_values)
from epymorph.util import NDIndices, as_sorted_dict


class StaticGeo(Geo[StaticGeoSpec]):
    """A Geo implementation which contains all of data pre-fetched and in-memory."""

    values: dict[str, NDArray]

    def __init__(self, spec: StaticGeoSpec, values: dict[str, NDArray]):
        if not LABEL.name in values or not np.issubdtype(values[LABEL.name].dtype, np.str_):
            msg = "Geo must contain an attribute called 'label' of type string."
            raise ValueError(msg)
        self.values = values
        super().__init__(spec, len(values[LABEL.name]))

    def __getitem__(self, name: str) -> NDArray:
        if name not in self.values:
            raise AttributeException(f"Attribute not found in geo: '{name}'")
        return self.values[name]

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
        return self.__class__(self.spec, filtered_values)

    def save(self, file: os.PathLike) -> None:
        """Saves this geo to tar format."""
        StaticGeoFileOps.save_as_archive(self, file)


class StaticGeoFileOps:
    """Helper functions for saving and loading static geos as files."""

    @staticmethod
    def to_archive_filename(geo_id: str) -> str:
        """Returns the standard filename for a geo archive."""
        return f"{geo_id}.geo.tar"

    @staticmethod
    def to_geo_name(filename: str) -> str:
        """Returns the geo ID from a standard geo archive filename."""
        return filename.removesuffix('.geo.tar')

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
                if f.is_file() and f.name.endswith('.geo.tar'))

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
                if f.is_file() and f.name.endswith('.geo.tar'))

    @staticmethod
    def save_as_archive(geo: StaticGeo, file: os.PathLike) -> None:
        """Save a StaticGeo to its tar format."""

        # Write the data file in memory
        npz_file = io.BytesIO()
        # sorting the geo values makes the sha256 a little more stable
        np.savez_compressed(npz_file, **as_sorted_dict(geo.values))
        # Data checksum
        npz_file.seek(0)
        data_sha256 = hashlib.sha256()
        data_sha256.update(npz_file.read())

        # Write the spec file in memory
        geo_file = io.BytesIO()
        geo_json = cast(str, jsonpickle.encode(geo.spec, unpicklable=True))
        geo_file.write(geo_json.encode('utf-8'))
        # Spec checksum
        geo_file.seek(0)
        spec_sha256 = hashlib.sha256()
        spec_sha256.update(geo_file.read())

        # Write sha256 checksums file in memory
        sha_file = io.BytesIO()
        sha_text = f"""\
{data_sha256.hexdigest()}  data.npz
{spec_sha256.hexdigest()}  spec.geo"""
        sha_file.write(bytes(sha_text, encoding='utf-8'))

        # Write the tar to disk
        with tarfile.open(file, 'w') as tar:
            def add_file(contents: io.BytesIO, name: str) -> None:
                info = tarfile.TarInfo(name)
                info.size = contents.tell()
                contents.seek(0)
                tar.addfile(info, contents)

            add_file(npz_file, 'data.npz')
            add_file(geo_file, 'spec.geo')
            add_file(sha_file, 'checksums.sha256')

    @staticmethod
    def load_from_archive(file: os.PathLike) -> StaticGeo:
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
                sha_file = tar.extractfile(tar.getmember('checksums.sha256'))

                if npz_file is None or geo_file is None or sha_file is None:
                    msg = 'Archive is incomplete: missing data, spec, and/or checksum files.'
                    raise GeoValidationException(msg)

                # Verify the checksums
                for line_bytes in sha_file.readlines():
                    line = str(line_bytes, encoding='utf-8')
                    [checksum, filename] = line.strip().split('  ')
                    match filename:
                        case 'data.npz':
                            file_to_check = npz_file
                        case 'spec.geo':
                            file_to_check = geo_file
                        case _:
                            # There shouldn't be any other files listed in the checksum.
                            msg = f"Unknown file listing in checksums.sha256 ({filename})."
                            raise GeoValidationException(msg)
                    file_to_check.seek(0)
                    sha256 = hashlib.sha256()
                    sha256.update(file_to_check.read())
                    if checksum != sha256.hexdigest():
                        msg = f"Archive checksum did not match (for file {filename}). "\
                            "It is possible the file has been corrupted."
                        raise GeoValidationException(msg)

                # Read the spec file in memory
                geo_file.seek(0)
                spec_json = geo_file.read().decode('utf8')
                spec = StaticGeoSpec.deserialize(spec_json)

                # Read the data file in memory
                npz_file.seek(0)
                with np.load(npz_file) as data:
                    values = dict(data)

                return StaticGeo(spec, values)

        except Exception as e:
            raise GeoValidationException(f"Unable to load '{file}' as a geo.") from e
