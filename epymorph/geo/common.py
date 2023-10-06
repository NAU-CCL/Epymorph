"""Common utilities for the geo subsystem."""
from typing import NamedTuple

import numpy as np

from epymorph.util import DTLike

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])
"""Structured numpy dtype for long/lat coordinates."""


class AttribDef(NamedTuple):
    """Metadata about a Geo attribute."""
    name: str
    dtype: DTLike
