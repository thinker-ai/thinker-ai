from abc import ABC, abstractmethod

from thinker_ai.tasks.static.task_flow import TaskFlow


class TaskFlowFlowRepository(ABC):
    @abstractmethod
    def save(self, task_flow: TaskFlow):
        raise NotImplementedError

    @abstractmethod
    def load(self,id: str) -> TaskFlow:
        raise NotImplementedError
