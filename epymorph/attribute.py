from dataclasses import dataclass
from typing import Literal

import numpy as np

from epymorph.data_shape import DataShape

AttributeType = type[int | float | str]
"""All allowed attribute types (as Python types)."""

AttributeTypeNp = np.int64 | np.float64 | np.str_
"""All allowed attribute types (as numpy types)."""


@dataclass(frozen=True)
class AttributeDef:
    """Definition of a simulation attribute."""
    name: str
    shape: DataShape
    dtype: AttributeType
    source: Literal['geo', 'params']

    @property
    def dtype_as_np(self) -> np.dtype:
        if self.dtype == int:
            return np.dtype(np.int64)
        elif self.dtype == float:
            return np.dtype(np.float64)
        elif self.dtype == str:
            return np.dtype(np.str_)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
