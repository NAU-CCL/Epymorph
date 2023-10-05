from typing import NamedTuple

import numpy as np

from epymorph.util import DTLike

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])


class AttribDef(NamedTuple):
    """Metadata about a Geo attribute."""
    name: str
    dtype: DTLike
