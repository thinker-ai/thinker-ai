from abc import abstractmethod, ABC
from typing import Dict

from pydantic import BaseModel

from thinker_ai.actions.action import Criteria
from thinker_ai.context import Context


class Work(BaseModel, ABC):
    solution_id: str
    work_id: str
    agent_id:str
    criteria: Criteria

    def __init__(self, id: str,agent_id:str, name: str, context: Context, *args, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.agent_id=agent_id
        self.name = name
        self.context = context

    @abstractmethod
    async def do(self, *args, **kwargs):
        raise NotImplementedError


class CompositeWork(Work, ABC):
    tasks: Dict[str, Work] = {}

    def __init__(self, id: str, name: str, context: Context, *args, **kwargs):
        super().__init__(id, name, context, args, kwargs)

    def add_task(self, work: Work):
        self.tasks[work.work_id] = work
