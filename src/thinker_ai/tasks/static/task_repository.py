from abc import ABC, abstractmethod

from thinker_ai.tasks.static.task import Task


class TaskRepository(ABC):
    @abstractmethod
    def save(self,task: Task):
        raise NotImplementedError

    @abstractmethod
    def load(self,id:str)->Task:
        raise NotImplementedError
