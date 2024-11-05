from abc import ABC, abstractmethod


class Likelihood(ABC):
    def __init__(self) -> None:
        return None

    @abstractmethod
    def compute(self, observed: int, expected: int):
        raise NotImplementedError("Subclasses should implement this method.")
