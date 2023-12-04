from abc import abstractmethod, ABC
from typing import Dict

from pydantic import BaseModel

from thinker_ai.actions import Criteria
from thinker_ai.context import Context


class Task(BaseModel, ABC):
    solution_id: str
    task_id: str
    criteria: Criteria

    def __init__(self, id: str, name: str, context: Context, *args, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.name = name
        self.context = context

    @abstractmethod
    async def do(self, *args, **kwargs):
        raise NotImplementedError


class CompositeTask(Task, ABC):
    tasks: Dict[str, Task] = {}

    def __init__(self, id: str, name: str, context: Context, *args, **kwargs):
        super().__init__(id, name, context, args, kwargs)

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task
