from abc import ABC, abstractmethod

from thinker_ai.task_flow.task_flow import TaskFlow


class TaskFlowFlowRepository(ABC):
    @abstractmethod
    def save(self, task_flow: TaskFlow):
        raise NotImplementedError

    @abstractmethod
    def load(self,id: str) -> TaskFlow:
        raise NotImplementedError
