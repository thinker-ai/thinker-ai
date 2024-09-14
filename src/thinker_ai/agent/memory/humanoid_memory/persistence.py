from abc import ABC, abstractmethod
from typing import Any


class MemoryPersistence(ABC):
    """
    持久化接口，定义保存和加载的方法。
    """

    @abstractmethod
    def save(self, data: Any):
        pass

    @abstractmethod
    def load(self) -> Any:
        pass