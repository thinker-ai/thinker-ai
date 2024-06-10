from abc import ABC, abstractmethod

from thinker_ai.tasks.static.task_schema_schema import TaskFlowSchema


class TaskFlowSchemaRepository(ABC):
    @abstractmethod
    def save(self, task_schema: TaskFlowSchema):
        raise NotImplementedError

    @abstractmethod
    def load(self, id: str) -> TaskFlowSchema:
        raise NotImplementedError
