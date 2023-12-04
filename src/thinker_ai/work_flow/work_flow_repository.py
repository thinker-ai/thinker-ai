from abc import ABC, abstractmethod

from thinker_ai.work_flow.work_flow import WorkFlow


class WorkFlowRepository(ABC):
    @abstractmethod
    def save(self, task_flow: WorkFlow):
        raise NotImplementedError

    @abstractmethod
    def load(self,id: str) -> WorkFlow:
        raise NotImplementedError
