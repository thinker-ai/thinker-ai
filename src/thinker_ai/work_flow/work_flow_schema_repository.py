from abc import ABC, abstractmethod

from thinker_ai.work_flow.work_schema_schema import WorkFlowSchema


class WorkFlowSchemaRepository(ABC):
    @abstractmethod
    def save(self, task_schema: WorkFlowSchema):
        raise NotImplementedError

    @abstractmethod
    def load(self, id: str) -> WorkFlowSchema:
        raise NotImplementedError
