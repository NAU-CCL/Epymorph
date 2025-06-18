"""
Our custom pytest configuration.

Most of the functionality in this file enables us to use VCR to record HTTP
request/response activity during tests.
"""

import gzip
from contextlib import nullcontext
from pathlib import Path

import pytest
from vcr import VCR
from vcr.config import RecordMode
from vcr.persisters.filesystem import CassetteDecodeError, CassetteNotFoundError
from vcr.serialize import deserialize, serialize


class GzipPersister:
    """Persist VCR cassettes as gzipped files."""

    @classmethod
    def load_cassette(cls, cassette_path, serializer):
        try:
            cassette_path = Path(cassette_path)
            if not cassette_path.is_file():
                raise CassetteNotFoundError()
            with cassette_path.open("rb") as f:
                data = gzip.decompress(f.read()).decode("utf-8")
        except UnicodeDecodeError as e:
            err = "Unable to decode cassette content as utf-8."
            raise CassetteDecodeError(err) from e
        return deserialize(data, serializer)

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        data = serialize(cassette_dict, serializer)
        cassette_path = Path(cassette_path)
        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        with cassette_path.open("wb") as f:
            f.write(gzip.compress(data.encode("utf-8")))


class OverwriteGzipPersister:
    @classmethod
    def load_cassette(cls, cassette_path, serializer):
        # By simply refusing to load the existing cassette whether it exists or not,
        # VCR will fall back to issuing real HTTP requests and (if the mode is right)
        # record them. Recording truncates any existing file before write.
        raise CassetteNotFoundError()

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        GzipPersister.save_cassette(cassette_path, cassette_dict, serializer)


# Pytest option `--vcr-mode`:
# Use this option from the pytest command line to set VCR recording behavior.
# NOTE: the options I'm supporting are different from the vcrpy's options.
# The built-in options are kinda confusing and insufficient besides.
#
# Our options are:
# - auto: (default) record new cassettes, replay old cassettes
# - replay: replay cassettes only, no recording
# - rerecord: delete and re-record all cassettes, no replaying
#
# Implementation: "replay" acts like vcrpy's "none" mode;
# "rerecord" acts like "once" mode but it first deletes existing cassettes,
# effectively overwriting them.
_RECORD_MODE_KEY = "--vcr-mode"
_RECORD_MODE_VALUES = ("auto", "replay", "rerecord")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        _RECORD_MODE_KEY,
        action="store",
        default="auto",
        choices=_RECORD_MODE_VALUES,
        help="VCR record mode.",
    )


# Marker `vcr`:
# Mark a test with @pytest.mark.vcr and
# it will automatically run in the VCR.use_cassette() context
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "vcr: use VCR.py during the test.")


# Fixture `vcr_config`:
# Modules can add/override VCR constructor options by overridding this fixture.
# The value should be a dict, whose items will be merged with the VCR init kwargs.
@pytest.fixture(scope="module")
def vcr_config():
    return {}


# Fixture `vcr`:
# The VCR instance to use during a test. Tests shouldn't use this fixture directly;
# instead just mark tests with the vcr marker and our wrapper hook will automatically
# run the test in the VCR use_cassettes() context.
@pytest.fixture(scope="module", autouse=True)
def vcr(request, vcr_config):
    record_mode = request.config.getoption(_RECORD_MODE_KEY)
    match record_mode:
        case "auto":
            vcrpy_mode = RecordMode.NEW_EPISODES
            persister = GzipPersister()
        case "replay":
            vcrpy_mode = RecordMode.NONE
            persister = GzipPersister()
        case "record":
            vcrpy_mode = RecordMode.ONCE
            persister = OverwriteGzipPersister()
        case _:
            err = f"Unknown --vcr-mode: '{record_mode}'"
            raise ValueError(err)

    # Always drop response headers in VCR cassettes.
    # We don't need them, they're verbose, and they often contain sensitive info.
    def drop_resp_headers(response):
        response["headers"] = {}
        return response

    vcr = VCR(
        record_mode=vcrpy_mode,
        before_record_response=drop_resp_headers,
        **(vcr_config or {}),
    )
    vcr.register_persister(persister)
    return vcr


# Wrap the pytest_runtest_call hook:
# run tests in a VCR.use_cassette() context if test is marked `vcr`.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    cassette_path = (
        item.path.parent / "cassettes" / item.path.stem / f"{item.name}.yaml.gz"
    )
    vcr = item.funcargs["vcr"]  # type: ignore
    context = (
        vcr.use_cassette(cassette_path)
        if item.get_closest_marker("vcr")
        else nullcontext()
    )
    with context:
        yield
