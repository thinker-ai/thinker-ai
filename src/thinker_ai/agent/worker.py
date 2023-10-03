from abc import abstractmethod
from typing import Dict

from thinker_ai.actions.action import Criteria, BaseAction
from thinker_ai.agent.agent import Agent
from thinker_ai.task.task import Task


class Worker(Agent):
    def __init__(self, name: str, criteria: Criteria, actions: Dict[str, BaseAction]):
        super().__init__(name, actions)
        self.criteria = criteria

    @abstractmethod
    async def work(self, task: Task, *args, **kwargs):
        raise NotImplementedError