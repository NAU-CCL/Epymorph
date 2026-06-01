#!/bin/bash
set -euo pipefail

test_version () {
    py=$1
    VIRTUAL_ENV=".venv-$py" uv run \
        --active --python "$py" --no-cache \
        --no-build --no-binary-package epymorph \
        pytest tests/fast --quiet
}

test_version 3.11
test_version 3.12
test_version 3.13
test_version 3.14
