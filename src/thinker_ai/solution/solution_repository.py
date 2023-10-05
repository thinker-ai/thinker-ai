from abc import ABC, abstractmethod

from thinker_ai.solution.solution import Solution


class SolutionRepository(ABC):
    @abstractmethod
    def save(self, solution: Solution):
        raise NotImplementedError

    @abstractmethod
    def load(self,id: str) -> Solution:
        raise NotImplementedError
