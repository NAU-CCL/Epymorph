from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from requests import Response

from epymorph import cache

URL = "https://example.com/data.zip"
CACHE_KEY = Path("data/data.zip")
VALID_CONTENT = b"valid file contents"


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path)
    monkeypatch.delenv("EPYMORPH_CACHE_DISABLED", raising=False)
    monkeypatch.delenv("EPYMORPH_CACHE_DISABLED_PATHS", raising=False)
    return tmp_path


def test_load_or_fetch_returns_bytes_by_default(cache_root):
    result = cache.load_or_fetch(
        CACHE_KEY,
        lambda: BytesIO(VALID_CONTENT),
    )

    assert isinstance(result, BytesIO)
    assert result.read() == VALID_CONTENT


def test_load_or_fetch_caches_original_bytes_after_mapping(cache_root):
    result = cache.load_or_fetch(
        CACHE_KEY,
        lambda: BytesIO(VALID_CONTENT),
        map_fn=lambda file: file.read().decode(),
    )

    assert result == VALID_CONTENT.decode()
    assert (cache_root / CACHE_KEY).read_bytes() == VALID_CONTENT


def test_load_or_fetch_does_not_cache_when_mapping_fails(cache_root):
    error = ValueError("Unexpected file contents")
    mapper = Mock(side_effect=error)

    with pytest.raises(cache.FileReadError, match="Unable to read file") as exc:
        cache.load_or_fetch(
            CACHE_KEY,
            lambda: BytesIO(b"Request Rejected"),
            map_fn=mapper,
        )

    assert exc.value.__cause__ is error
    assert not (cache_root / CACHE_KEY).exists()


def test_load_or_fetch_maps_cached_bytes_without_fetching(cache_root):
    destination = cache_root / CACHE_KEY
    destination.parent.mkdir()
    destination.write_bytes(VALID_CONTENT)
    fetch = Mock()

    result = cache.load_or_fetch(
        CACHE_KEY,
        fetch,
        map_fn=lambda file: file.read().decode(),
    )

    assert result == VALID_CONTENT.decode()
    fetch.assert_not_called()


def test_load_or_fetch_does_not_save_when_cache_is_disabled(cache_root, monkeypatch):
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    cache.load_or_fetch(CACHE_KEY, lambda: BytesIO(VALID_CONTENT))

    assert not (cache_root / CACHE_KEY).exists()


def test_load_or_fetch_url_maps_response_bytes(cache_root, monkeypatch):
    response = Response()
    response.status_code = 200
    response.raw = BytesIO(VALID_CONTENT)
    monkeypatch.setattr(cache.requests, "get", Mock(return_value=response))

    result = cache.load_or_fetch_url(
        URL,
        CACHE_KEY,
        map_fn=lambda file: file.read(),
    )

    assert result == VALID_CONTENT
