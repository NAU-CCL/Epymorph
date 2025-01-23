from abc import ABC, abstractmethod


class Likelihood(ABC):
    @abstractmethod
    def compute(self, observed: int, expected: int):
        raise NotImplementedError("Subclasses should implement this method.")
