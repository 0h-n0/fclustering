from abc import ABC, abstractmethod


class Download(ABC):
    @abstractmethod
    def download(self):
        raise NotImplementedError()

        

