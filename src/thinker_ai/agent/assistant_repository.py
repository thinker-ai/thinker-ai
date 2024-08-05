from abc import ABC, abstractmethod
from typing import Optional, List

from thinker_ai.agent.assistant import AssistantInterface
from thinker_ai.configs.const import PROJECT_ROOT


class AssistantRepository(ABC):

    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT / 'data/agents.q') -> "AssistantRepository":
        raise NotImplementedError
    @abstractmethod
    def add_assistant(self, assistant: AssistantInterface, user_id: str):
        raise NotImplementedError
    @abstractmethod
    def get_assistant(self, assistant_id: str, user_id: str) -> Optional[AssistantInterface]:
        raise NotImplementedError
    @abstractmethod
    def get_my_assistant_ids(self, user_id) -> List[str]:
        raise NotImplementedError
    @abstractmethod
    def update_assistant(self, assistant: AssistantInterface, user_id: str):
        raise NotImplementedError
    @abstractmethod
    def delete_assistant(self, assistant_id: str, user_id: str):
        raise NotImplementedError
    @classmethod
    @abstractmethod
    def reset_instance(cls):
        raise NotImplementedError
