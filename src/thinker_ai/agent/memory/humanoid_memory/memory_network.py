from abc import ABC, abstractmethod
from typing import List


class MemoryNetwork(ABC):
    @abstractmethod
    def add_memory(self, inputs: List[str]):
        """
        将新的信息添加到记忆网络中。
        """
        pass
    @abstractmethod
    def query(self, question: str) -> str:
        """
        根据输入的问题，在记忆网络中检索相关的答案或信息。
        """
        pass
    @abstractmethod
    def is_related(self, text1: str, text2: str, similarity_threshold: float)-> bool:
        """
        判断两个文本在记忆网络中是否相关。
        """
        pass
    @abstractmethod
    def clear_memory(self):
        """
        清除记忆网络中的所有信息。
        """
        pass
    @abstractmethod
    def save(self, data):
        pass
    @abstractmethod
    def load(self):
        pass