from threading import Lock
from typing import Optional, List, Literal

from thinker_ai.agent.assistant_api import AssistantApi
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.openai_assistant_api import openai_client
from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from thinker_ai.agent.openai_assistant_api.openai_assistant_api_dao import OpenAiAssistantApiDAO, OpenAiAssistantApiPO
from thinker_ai.configs.const import PROJECT_ROOT


class OpenAiAssistantRepository(AssistantRepository):
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, filepath: Optional[
        str] = PROJECT_ROOT / 'data/test_assistants.json') -> "OpenAiAssistantRepository":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(filepath)
        return cls._instance

    def __init__(self, filepath: str):
        if not OpenAiAssistantRepository._instance:
            self.filepath = filepath
            self.assistant_api_dao = OpenAiAssistantApiDAO(filepath)
            OpenAiAssistantRepository._instance = self
        else:
            raise Exception("Attempting to instantiate a singleton class.")

    def add_assistant_api(self, user_id: str, assistant_api: OpenAiAssistantApi):
        if assistant_api.user_id != user_id:
            raise PermissionError("Cannot add an agent for another user.")
        with self._lock:
            assistant_api_po = OpenAiAssistantApiPO.from_assistant_api(assistant_api)
            self.assistant_api_dao.add_assistant_api(assistant_api_po)

    def get_assistant_api_by_id(self, user_id: str, assistant_id: str) -> Optional[AssistantApi]:
        with self._lock:
            assistant_api_po = self.assistant_api_dao.get_assistant_api_by_id(assistant_id)
            if assistant_api_po is not None and assistant_api_po.user_id == user_id:
                return assistant_api_po.to_assistant_api()
            return None

    def get_all_assistant_api_of(self, user_id, of: Literal["name", "assistant_id"]) -> List[str]:
        with self._lock:
            return self.assistant_api_dao.get_my_assistants_of(user_id=user_id, of=of)

    def update_assistant_name(self, user_id: str, assistant_id: str, name: str):
        with self._lock:
            openai_client.beta.assistants.update(assistant_id=assistant_id, name=name)
            self.assistant_api_dao.update_assistant_name(user_id=user_id,assistant_id=assistant_id, name=name)

    def update_topic_name(self, user_id: str, assistant_id: str, old_topic: str, new_topic: str):
        with self._lock:
            self.assistant_api_dao.update_topic_name(user_id=user_id,assistant_id=assistant_id,
                                                     old_topic=old_topic,new_topic=new_topic)

    def get_assistant_api_by_name(self, user_id: str, name: str) -> Optional[AssistantApi]:
        with self._lock:
            assistant_api_po = self.assistant_api_dao.get_assistant_api_by_name(name=name, user_id=user_id)
            if assistant_api_po is not None:
                return assistant_api_po.to_assistant_api()
            return None

    def delete_assistant_api(self, user_id: str, assistant_id: str):
        agent = self.get_assistant_api_by_id(user_id=user_id, assistant_id=assistant_id)
        if agent is None:
            raise PermissionError("Cannot delete an agent for another user.")
        with self._lock:
            self.assistant_api_dao.delete_assistant_api(assistant_id)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
