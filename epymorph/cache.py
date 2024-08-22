"""epymorph's file caching utilities."""
from hashlib import sha256
from io import BytesIO
from math import log
from os import PathLike, getenv
from pathlib import Path
from shutil import rmtree
from tarfile import TarInfo, is_tarfile
from tarfile import open as open_tarfile
from typing import Callable, NamedTuple, Sequence
from urllib.request import urlopen
from warnings import warn

from platformdirs import user_cache_path


def _cache_path() -> Path:
    if (path_var := getenv("EPYMORPH_CACHE_PATH")) is not None:
        # Load path from env var
        path = Path(path_var)
    else:
        # fall back to platform-specific default path
        path = user_cache_path(appname='epymorph')
    # ensure cache directory exists
    path.mkdir(parents=True, exist_ok=True)
    return path


CACHE_PATH = _cache_path()


class FileError(Exception):
    """Error during a file operation."""


class FileMissingError(FileError):
    """Error loading a file, as it does not exist."""


class FileWriteError(FileError):
    """Error writing a file."""


class FileReadError(FileError):
    """Error loading a file."""


class FileVersionError(FileError):
    """Error loading a file due to unmet version requirements."""


class CacheMiss(FileError):
    """Raised on a cache-miss (for any reason) during a load-from-cache operation."""


class CacheWarning(Warning):
    """
    Warning issued when we are unable to interact with the file cache but in a situation where 
    program execution can continue, even if less optimally. For example: if we successfully load data
    from an external source but are unable to cache it for later, this is a warning because we assume
    the data is valid and that it could always be loaded again from the same source at a later time.
    The warning is issued to give the user the opportunity to fix it for next time.
    """


def save_file(to_path: str | PathLike[str], file: BytesIO) -> None:
    """
    Save a single file. `to_path` can be absolute or relative; relative paths will be resolved
    against the current working directory. Folders in the path which do not exist will be created
    automatically.
    """
    try:
        file_path = Path(to_path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode="wb") as f:
            f.write(file.getbuffer())
    except Exception as e:
        msg = f"Unable to write file at path: {to_path}"
        raise FileWriteError(msg) from e


def load_file(from_path: str | PathLike[str]) -> BytesIO:
    """
    Load a single file. An Exception is raised if the file cannot be loaded for any reason.
    On success, returns the bytes of the file.
    """
    try:
        file_path = Path(from_path).resolve()
        if not file_path.is_file():
            raise FileMissingError(f"No file at: {file_path}")

        # Read the file into memory
        file_buffer = BytesIO()
        with open(file_path, 'rb') as f:
            file_buffer.write(f.read())
        file_buffer.seek(0)
        return file_buffer
    except FileError:
        raise
    except Exception as e:
        raise FileReadError(f"Unable to load file at: {from_path}") from e


def save_bundle(to_path: str | PathLike[str], version: int, files: dict[str, BytesIO]) -> None:
    """
    Save a bundle of files in our tar format with an associated version number.
    `to_path` can be absolute or relative; relative paths will be resolved
    against the current working directory. Folders in the path which do not exist
    will be created automatically.
    """

    if version <= 0:
        raise ValueError("version should be greater than zero.")

    try:
        # Compute checksums
        sha_entries = []
        for name, contents in files.items():
            contents.seek(0)
            sha = sha256()
            sha.update(contents.read())
            sha_entries.append(f"{sha.hexdigest()}  {name}")

        # Create checksums.sha256 file
        sha_file = BytesIO()
        sha_text = "\n".join(sha_entries)
        sha_file.write(bytes(sha_text, encoding='utf-8'))

        # Create cache version file
        ver_file = BytesIO()
        ver_file.write(bytes(str(version), encoding="utf-8"))

        tarred_files = {
            **files,
            "checksums.sha256": sha_file,
            "version": ver_file,
        }

        # Write the tar to disk
        tar_path = Path(to_path).resolve()
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'w:gz' if tar_path.suffix == '.tgz' else 'w'
        with open_tarfile(name=tar_path, mode=mode) as tar:
            for name, contents in tarred_files.items():
                info = TarInfo(name)
                info.size = contents.tell()
                contents.seek(0)
                tar.addfile(info, contents)

    except Exception as e:
        msg = f"Unable to write archive at path: {to_path}"
        raise FileWriteError(msg) from e


def load_bundle(from_path: str | PathLike[str], version_at_least: int = -1) -> dict[str, BytesIO]:
    """
    Load a bundle of files in our tar format, optionally enforcing a minimum version.
    An Exception is raised if the file cannot be loaded for any reason, or if its version
    is incorrect. On success, returns a dictionary of the contained files, mapping the file
    name to the bytes of the file.
    """
    try:
        tar_path = Path(from_path).resolve()
        if not tar_path.is_file():
            raise FileMissingError(f"No file at: {tar_path}")

        # Read the tar file into memory
        tar_buffer = BytesIO()
        with open(tar_path, 'rb') as f:
            tar_buffer.write(f.read())
        tar_buffer.seek(0)

        if not is_tarfile(tar_buffer):
            raise FileReadError(f"Not a tar file at: {tar_path}")

        mode = 'r:gz' if tar_path.suffix == '.tgz' else 'r'
        tarred_files: dict[str, BytesIO] = {}
        with open_tarfile(fileobj=tar_buffer, mode=mode) as tar:
            for info in tar.getmembers():
                name = info.name
                contents = tar.extractfile(info)
                if contents is not None:
                    tarred_files[name] = BytesIO(contents.read())

        # Check version
        if "version" in tarred_files:
            ver_file = tarred_files["version"]
            version = int(str(ver_file.readline(), encoding="utf-8"))
        else:
            version = -1
        if version < version_at_least:
            raise FileVersionError("Archive is an unacceptable version.")

        # Verify the checksums
        if "checksums.sha256" not in tarred_files:
            raise FileReadError("Archive appears to be invalid.")
        sha_file = tarred_files["checksums.sha256"]
        for line_bytes in sha_file.readlines():
            line = str(line_bytes, encoding='utf-8')
            [checksum, filename] = line.strip().split('  ')

            if filename not in tarred_files:
                raise FileReadError("Archive appears to be invalid.")

            contents = tarred_files[filename]
            contents.seek(0)
            sha = sha256()
            sha.update(contents.read())
            contents.seek(0)
            if checksum != sha.hexdigest():
                msg = f"Archive checksum did not match (for file {filename}). "\
                    "It is possible the file is corrupt."
                raise FileReadError(msg)

        return {
            name: contents
            for name, contents in tarred_files.items()
            if name not in ("checksums.sha256", "version")
        }

    except FileError:
        raise
    except Exception as e:
        raise FileReadError(f"Unable to load archive at: {from_path}") from e


def _resolve_cache_path(path: str | PathLike[str]) -> Path:
    cache_path = Path(path)
    if cache_path.is_absolute():
        raise ValueError(
            "When saving to or loading from the cache, please supply a relative path."
        )
    resolved = CACHE_PATH.joinpath(cache_path).resolve()
    if not resolved.is_relative_to(CACHE_PATH):
        # Ensure the resolved path is still inside CACHE_PATH.
        raise ValueError(
            "When saving to or loading from the cache, please supply a relative path."
        )
    return resolved


def save_file_to_cache(to_path: str | PathLike[str], file: BytesIO) -> None:
    """
    Save a single file to the cache (overwriting the existing file, if any).
    This is a low-level building block.
    """
    save_file(_resolve_cache_path(to_path), file)


def load_file_from_cache(from_path: str | PathLike[str]) -> BytesIO:
    """
    Load a single file from the cache.
    This is a low-level building block.
    """
    try:
        return load_file(_resolve_cache_path(from_path))
    except FileError as e:
        raise CacheMiss() from e


def load_or_fetch(cache_path: Path, fetch: Callable[[], BytesIO]) -> BytesIO:
    """
    Attempts to load a file from the cache. If it doesn't exist, uses the provided
    fetch method to load the file, then attempts to save the file to the cache for 
    next time. (This is a higher-level but still generic building block.)
    Any exceptions raised by `fetch` will not be caught in this method.
    """
    try:
        # Try to load from cache.
        return load_file_from_cache(cache_path)
    except CacheMiss:
        # On cache miss, fetch file contents.
        file = fetch()
        # And attempt to save the file to the cache for next time.
        try:
            save_file_to_cache(cache_path, file)
        except Exception as e:
            # Failure to save to the cache is not worth stopping the program:
            # raise a warning.
            warn(
                f"Unable to save file to the cache ({cache_path}). Cause:\n{e}",
                CacheWarning
            )
        return file


def load_or_fetch_url(url: str, cache_path: Path) -> BytesIO:
    """
    Attempts to load a file from the cache. If it doesn't exist, fetches
    the file contents from the given URL, then attempts to save the file to the cache
    for next time.
    """
    def fetch_url():
        with urlopen(url) as f:
            return BytesIO(f.read())
    return load_or_fetch(cache_path, fetch_url)


def save_bundle_to_cache(to_path: str | PathLike[str], version: int, files: dict[str, BytesIO]) -> None:
    """
    Save a tar bundle of files to the cache (overwriting the existing file, if any).
    The tar includes the sha256 checksums of every content file,
    and a version file indicating which application version was
    responsible for writing the file (thus allowing the application
    to decide if a cached file is still valid when reading it).
    """
    save_bundle(_resolve_cache_path(to_path), version, files)


def load_bundle_from_cache(from_path: str | PathLike[str], version_at_least: int = -1) -> dict[str, BytesIO]:
    """
    Load a tar bundle of files from the cache. `from_path` must be a relative path.
    `version_at_least` optionally specifies a version number that must be met or beat
    by the cached file in order for the file to be considered valid. If the cached file
    was written against a version less than this, it will be considered a cache miss
    (raises CacheMiss).
    """
    try:
        return load_bundle(_resolve_cache_path(from_path), version_at_least)
    except FileError as e:
        raise CacheMiss() from e


####################
# Cache Management #
####################


# https://en.wikipedia.org/wiki/Metric_prefix
_suffixes = ('B', 'kiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB', 'RiB', 'QiB')


def format_file_size(size: int) -> str:
    """Format a file size given in bytes in 1024-based-unit representation."""
    if size < 0:
        raise ValueError("size cannot be less than zero.")
    if size < 1024:
        return f"{size} {_suffixes[0]}"
    magnitude = int(log(size, 1024))
    if magnitude >= len(_suffixes):
        raise ValueError("size is too large to format.")
    fsize = size / pow(1024, magnitude)
    return f"{fsize:.1f} {_suffixes[magnitude]}"


class Directory(NamedTuple):
    """A directory."""
    name: str
    """The directory name."""
    size: int
    """The combined size of all of this directory's children."""
    children: "Sequence[FileTree]"
    """The directory's children, which may be files or nested directories."""


class File(NamedTuple):
    """A file."""
    name: str
    """The file name."""
    size: int
    """The file size."""


FileTree = Directory | File
"""Nodes in a file tree are either directories or files."""


def cache_inventory() -> Directory:
    """Lists the contents of epymorph's cache as a FileTree."""
    def recurse(directory: Path) -> Directory:
        children = []
        size = 0
        for path in directory.iterdir():
            if path.is_symlink():
                # Ignore symlinks.
                continue
            if path.is_file():
                file_size = path.stat().st_size
                children.append(File(path.name, file_size))
                size += file_size
            elif path.is_dir():
                d = recurse(path)
                children.append(d)
                size += d.size
        return Directory(directory.name, size, children)

    if not CACHE_PATH.exists():
        return Directory(CACHE_PATH.name, 0, [])
    return recurse(CACHE_PATH)


def cache_remove_confirmation(path: str | PathLike[str]) -> tuple[Path, Callable[[], None]]:
    """
    Creates a function which removes a directory or file from the cache.
    Also returns the resolved path to the thing that will be removed;
    this allows the application to confirm the removal.
    """
    try:
        # This makes sure we don't delete things outside of the cache path.
        to_remove = _resolve_cache_path(path)
    except ValueError as e:
        raise FileError(str(e)) from None
    if not to_remove.exists():
        raise FileError(f"Given path is not in the cache: {to_remove}")

    def confirm_remove() -> None:
        # Remove the target file/dir
        if to_remove.is_file():
            to_remove.unlink()
        else:
            rmtree(to_remove)

        # Remove any newly-empty parent directories, up to the cache dir
        parents = [p for p in to_remove.parents
                   if p.is_relative_to(CACHE_PATH)
                   and p != CACHE_PATH]
        for p in parents:
            if any(p.iterdir()):
                break  # parent not empty, we can stop
            p.rmdir()  # parent is empty

        # We may need to replace the cache dir if we just deleted it.
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return to_remove, confirm_remove


def cache_remove(path: str | PathLike[str]) -> None:
    """Removes a directory or file from the cache."""
    # This is the "no confirmation" version of `cache_remove_confirmation`
    _, confirm_remove = cache_remove_confirmation(path)
    confirm_remove()
