"""
A place to put pytest configuration for the 'slow' tests package.
"""

import pytest


def _drop_resp_headers(response):
    # For VCR cassettes: drop response headers.
    # We don't need them, they're verbose, and they might contain sensitive info.
    response["headers"] = {}
    return response


@pytest.fixture(scope="package")
def global_vcr_config():
    # NOTE: this isn't a fixture that pytest-recording uses;
    # for it to take effect, your package should supply the `vcr_config` fixture,
    # taking this as an arg, and extend it or return it unchanged.
    return {"before_record_response": _drop_resp_headers}
