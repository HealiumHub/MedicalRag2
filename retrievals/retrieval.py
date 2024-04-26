import abc


class Retrieval(abc.ABC):
    @abc.abstractmethod
    def search(self, queries: list[str] = []) -> list:
        pass
