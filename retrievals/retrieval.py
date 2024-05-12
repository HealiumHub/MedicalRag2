import abc

from models.types import Source


class Retrieval(abc.ABC):
    @abc.abstractmethod
    def search(self, queries: list[str] = []) -> list[Source]:
        pass
