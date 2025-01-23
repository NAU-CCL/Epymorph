from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def rvs(self, size: int):
        raise NotImplementedError("Subclasses should implement this method.")
