# def format_size(size: int) -> str:
#     """
#     Given a file size in bytes, produce a 1024-based unit representation
#     with the decimal in a consistent position, and padded with spaces as necessary.
#     """
#     if abs(size) < 1024:
#         return f"{size:3d}.  "

#     fnum = float(size)
#     magnitude = 0
#     while abs(fnum) > 1024:
#         magnitude += 1
#         fnum = int(fnum / 100.0) / 10.0
#     suffix = [' B', ' kiB', ' MiB', ' GiB'][magnitude]
#     return f"{fnum:.1f}{suffix}"


# def get_total_size() -> str:
#     """Returns the total size of all files in the geo cache using 1024-based unit representation."""
#     total_size = sum((os.path.getsize(CACHE_PATH / file)
#                       for file, _ in F.iterate_dir_path(CACHE_PATH)))
#     return format_size(total_size)

# class StaticGeoFileOps:
#     """Helper functions for saving and loading static geos as files."""

#     @staticmethod
#     def to_archive_filename(geo_id: str) -> str:
#         """Returns the standard filename for a geo archive."""
#         return f"{geo_id}.geo.tgz"

#     @staticmethod
#     def to_geo_name(filename: str) -> str:
#         """Returns the geo ID from a standard geo archive filename."""
#         return filename.removesuffix('.geo.tgz')

#     @staticmethod
#     def iterate_dir(directory: Traversable) -> Iterator[tuple[Traversable, str]]:
#         """
#         Iterates through the given directory non-recursively, returning all archived geos.
#         Each item in the returned iterator is a tuple containing:
#         1. the Traversable instance for the file itself, and
#         2. the geo's ID.
#         """
#         return ((f, StaticGeoFileOps.to_geo_name(f.name))
#                 for f in directory.iterdir()
#                 if f.is_file() and f.name.endswith('.geo.tgz'))

#     @staticmethod
#     def iterate_dir_path(directory: Path) -> Iterator[tuple[Path, str]]:
#         """
#         Iterates through the given directory non-recursively, returning all archived geos.
#         Each item in the returned iterator is a tuple containing:
#         1. the Path for the file itself, and
#         2. the geo's ID.
#         """
#         return ((f, StaticGeoFileOps.to_geo_name(f.name))
#                 for f in directory.iterdir()
#                 if f.is_file() and f.name.endswith('.geo.tgz'))

#     @staticmethod
#     def save_as_archive(geo: StaticGeo, file: PathLike) -> None:
#         """Save a StaticGeo to its tar format."""

#         # Write the data file
#         # (sorting the geo values makes the sha256 a little more stable)
#         npz_file = BytesIO()
#         np.savez_compressed(npz_file, **as_sorted_dict(geo.values))

#         # Write the spec file
#         geo_file = BytesIO()
#         geo_json = cast(str, json_encode(geo.spec, unpicklable=True))
#         geo_file.write(geo_json.encode('utf-8'))

#         save_bundle(
#             to_path=file,
#             version=_STATIC_GEO_CACHE_VERSION,
#             files={
#                 "data.npz": npz_file,
#                 "spec.geo": geo_file,
#             },
#         )

#     @staticmethod
#     def load_from_archive(file: PathLike) -> StaticGeo:
#         """Load a StaticGeo from its tar format."""
#         try:
#             files = load_bundle(file, version_at_least=_STATIC_GEO_CACHE_VERSION)
#             if "data.npz" not in files or "spec.geo" not in files:
#                 msg = 'Archive is incomplete: missing data, spec, and/or checksum files.'
#                 raise GeoValidationException(msg)

#             # Read the spec file
#             geo_file = files["spec.geo"]
#             geo_file.seek(0)
#             spec_json = geo_file.read().decode('utf8')
#             spec = StaticGeoSpec.deserialize(spec_json)

#             # Read the data file
#             npz_file = files["data.npz"]
#             npz_file.seek(0)
#             with np.load(npz_file) as data:
#                 values = dict(data)

#             return StaticGeo(spec, values)
#         except Exception as e:
#             raise GeoValidationException(f"Unable to load '{file}' as a geo.") from e
