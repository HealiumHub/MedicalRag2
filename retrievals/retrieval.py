import abc


class Retrieval(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str) -> list:
        pass
