from abc import ABC, abstractmethod
from typing import Optional, List, Literal

from thinker_ai.agent.assistant_api import AssistantApi
from thinker_ai.configs.const import PROJECT_ROOT


class AssistantRepository(ABC):

    @classmethod
    def get_instance(cls,
                     filepath: Optional[str] = PROJECT_ROOT / 'data/test_assistants.json') -> "AssistantRepository":
        raise NotImplementedError

    @abstractmethod
    def add_assistant_api(self, user_id: str,assistant_api: AssistantApi):
        raise NotImplementedError

    @abstractmethod
    def get_assistant_api_by_id(self, user_id: str, assistant_id: str) -> Optional[AssistantApi]:
        raise NotImplementedError

    @abstractmethod
    def get_assistant_api_by_name(self, user_id: str, name: str) -> Optional[AssistantApi]:
        raise NotImplementedError

    @abstractmethod
    def get_all_assistant_api_of(self, user_id, of: Literal["name", "assistant_id"]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def update_assistant_name(self, user_id: str, assistant_id: str, name):
        raise NotImplementedError

    @abstractmethod
    def update_topic_name(self, user_id: str, assistant_id: str,old_topic:str, new_name:str):
        raise NotImplementedError

    @abstractmethod
    def delete_assistant_api(self, user_id: str, assistant_id: str, ):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reset_instance(cls):
        raise NotImplementedError
