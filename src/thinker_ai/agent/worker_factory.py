from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from pydantic import BaseModel

from thinker_ai.actions.action import BaseAction, Criteria
from thinker_ai.agent.agent import Worker


class WorkerConfig(BaseModel):
    name: str
    module_name: str
    class_name: str
    actions: Dict[str, BaseAction] = {}
    criteria: Criteria

class WorkerFactory(ABC):

    role_configs: Dict[str, WorkerConfig]

    def __init__(self, role_configs: List[WorkerConfig]):
        self.role_configs = {item.name: item for item in role_configs}

    def get_worker(self, name: str) -> Optional[Worker]:
        role_config = self.role_configs.get(name)
        if role_config:
            module = __import__(role_config.module_name, fromlist=[role_config.class_name])
            cls = getattr(module, role_config.class_name)
            instance = cls(name=role_config.name,criteria=role_config.criteria,actions=role_config.actions)
            return instance
        else:
            return None



