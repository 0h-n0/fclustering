from abc import ABC, abstractmethod

class Preprocess(ABC):
    @abstractmethod
    def preprocess(self):
        raise NotImplementedError()