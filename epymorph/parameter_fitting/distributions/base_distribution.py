from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def rvs(self, size: int, random_state: np.random.Generator):
        raise NotImplementedError("Subclasses should implement this method.")
