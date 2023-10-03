from abc import ABC, abstractmethod

from thinker_ai.context import Context
from thinker_ai.solution.solution import Solution
from thinker_ai.task.task import Task


class SolutionRepository(ABC):
    @abstractmethod
    def save(self, solution: Solution):
        raise NotImplementedError

    @abstractmethod
    def load(self,id: str) -> Solution:
        raise NotImplementedError
