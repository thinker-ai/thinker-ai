from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from thinker_ai.role.role import Role, RoleConfig


class RoleFactory(ABC):

    role_configs: Dict[str, RoleConfig]

    def __init__(self, role_configs: List[RoleConfig]):
        self.role_configs = {item.name: item for item in role_configs}

    def get_role(self, name: str) -> Optional[Role]:
        role_config = self.role_configs.get(name)
        if role_config:
            module = __import__(role_config.module_name, fromlist=[role_config.class_name])
            cls = getattr(module, role_config.class_name)
            instance = cls(role_config=role_config)
            return instance
        else:
            return None



