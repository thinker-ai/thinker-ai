from abc import abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel

from thinker_ai.actions.action import Criteria, BaseAction
from thinker_ai.agent.agent import Agent
from thinker_ai.solution.task import Task


class Worker(Agent):
    def __init__(self, name: str, criteria: Criteria, actions: Dict[str, BaseAction]):
        super().__init__(name, actions)
        self.criteria = criteria

    @abstractmethod
    async def work(self, task: Task, *args, **kwargs):
        raise NotImplementedError


class WorkerConfig(BaseModel):
    name: str
    module_name: str
    class_name: str
    actions: Dict[str, BaseAction] = {}
    criteria: Criteria


class WorkerFactory:
    role_configs: Dict[str, WorkerConfig]

    def __init__(self, role_configs: List[WorkerConfig]):
        self.role_configs = {item.name: item for item in role_configs}

    def get_worker(self, name: str) -> Optional[Worker]:
        role_config = self.role_configs.get(name)
        if role_config:
            module = __import__(role_config.module_name, fromlist=[role_config.class_name])
            cls = getattr(module, role_config.class_name)
            instance = cls(name=role_config.name, criteria=role_config.criteria, actions=role_config.actions)
            return instance
        else:
            return None
