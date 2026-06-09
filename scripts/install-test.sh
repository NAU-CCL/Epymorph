#!/bin/bash
#
# Tests that epymorph installs as a library without requiring users to build packages from source
# under a matrix of Python versions and dependency resolution strategies.
# 
# Importantly, this avoids using the uv.lock file to mimic end users installing from PyPI.
# Aside from testing installation itself, we also run the unit test suite just to verify basic functionality.
#
# Note that this creates and does not clean up a temporary directory.
#
# See .github/workflows/install-matrix-test.yml for a similar test that runs as a GHA across different OSes as well.
#
set -euo pipefail

export UV_EXCLUDE_NEWER="P7D"

epymorph_path=$(pwd)

tmp_dir=$(mktemp -d)
echo "Using temporary directory: $tmp_dir"

test_version() {
    dir_name=$1
    python_version=$2
    resolution=$3

    echo "=== Testing Python version $python_version ($resolution) ==="
    mkdir "$tmp_dir/$dir_name"
    cd "$tmp_dir/$dir_name"

    # These constraints are needed when using
    # `--resolution lowest` and `--no-build`
    cat > constraint.txt <<EOF
setuptools>=82
pillow>=12
pyproj>=3.7.2
kiwisolver>=1.4.9
fiona>=1.10
EOF

    # Set up a minimal cache -- just the geography summary files (.tgz's).
    # Otherwise we'll have to download a ton of data from TIGER and this will be really slow.
    # However if your system cache isn't already populated or is located somewhere else,
    # this might have no effect.
    mkdir -p .cache/epymorph/geography/us_tiger
    cp ~/.cache/epymorph/geography/us_tiger/*.tgz .cache/epymorph/geography/us_tiger
    export EPYMORPH_CACHE_PATH=$(realpath ./.cache/epymorph)

    uv init --bare --python $python_version
    uv python pin $python_version
    uv venv --managed-python
    uv add "$epymorph_path" --quiet \
        --no-build --no-binary-package epymorph \
        --resolution $resolution --constraint constraint.txt
    uv add pytest vcrpy --quiet
    uv run pytest "$epymorph_path/tests/fast" --quiet -W ignore::DeprecationWarning
    cd "$epymorph_path"
    echo "=== SUCCESS ==="
}

test_version py311-lo 3.11 lowest
test_version py311-hi 3.11 highest

test_version py312-lo 3.12 lowest
test_version py312-hi 3.12 highest

# NOTE: 3.13/3.14 not yet officially supported

# test_version py313-lo 3.13 lowest
# test_version py313-hi 3.13 highest

# test_version py314-lo 3.14 lowest
# test_version py314-hi 3.14 highest
