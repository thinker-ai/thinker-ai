from threading import Lock
from typing import Optional, List, cast

from thinker_ai.agent.assistant import AssistantInterface
from thinker_ai.agent.assistant_repository import AssistantRepository
from thinker_ai.agent.openai_assistant.openai_assistant import Assistant, OpenAiAssistant
from thinker_ai.agent.openai_assistant.openai_assistant_dao import OpenAiAssistantDAO, OpenAiAssistantPO
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.common.singleton_meta import SingletonMeta


class OpenAiAssistantRepository(AssistantRepository):
    _instance = None
    _lock = Lock()
    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT / 'data/agents.q') -> "OpenAiAssistantRepository":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(filepath)
        return cls._instance

    def __init__(self, filepath: str):
        if not OpenAiAssistantRepository._instance:
            self.filepath = filepath
            self.assistant_dao = OpenAiAssistantDAO(filepath)
            OpenAiAssistantRepository._instance = self
        else:
            raise Exception("Attempting to instantiate a singleton class.")

    def add_assistant(self, assistant: OpenAiAssistant, user_id: str):
        if assistant.user_id != user_id:
            raise PermissionError("Cannot add an agent for another user.")
        with self._lock:
            assistant_po = OpenAiAssistantPO.from_assistant(assistant)
            self.assistant_dao.add_assistant(assistant_po)

    def get_assistant(self, assistant_id: str, user_id: str) -> Optional[AssistantInterface]:
        with self._lock:
            assistant_po = self.assistant_dao.get_assistant(assistant_id)
            if assistant_po is not None and assistant_po.user_id == user_id:
                return assistant_po.to_assistant()
            return None

    def get_my_assistant_ids(self, user_id) -> List[str]:
        with self._lock:
            return self.assistant_dao.get_my_assistant_ids(user_id)

    def update_assistant(self, assistant: OpenAiAssistant, user_id: str):
        if assistant.user_id != user_id:
            raise PermissionError("Cannot update an agent for another user.")
        with self._lock:
            updated_assistant_po = OpenAiAssistantPO.from_assistant(assistant)
            self.assistant_dao.update_assistant(updated_assistant_po)

    def delete_assistant(self, assistant_id: str, user_id: str):
        agent = self.get_assistant(assistant_id, user_id)
        if agent is None:
            raise PermissionError("Cannot delete an agent for another user.")
        with self._lock:
            self.assistant_dao.delete_assistant(assistant_id)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
