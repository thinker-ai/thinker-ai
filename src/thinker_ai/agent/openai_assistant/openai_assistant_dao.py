import json
from threading import Lock
from typing import List, Optional

from thinker_ai.agent.openai_assistant import client
from thinker_ai.agent.openai_assistant.openai_assistant import OpenAiAssistant
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.common.serializable import Serializable
from thinker_ai.common.singleton_meta import SingletonMeta


class ThreadPO(Serializable):
    topic: str
    thread_id: str


class OpenAiAssistantPO(Serializable):
    user_id: str
    assistant_id: str
    topic_threads: List[ThreadPO]

    @classmethod
    def from_assistant(cls, assistant: OpenAiAssistant) -> "OpenAiAssistantPO":
        threads_po = [ThreadPO(topic=topic, thread_id=thread_id) for topic, thread_id in assistant.topic_threads.items()]
        return OpenAiAssistantPO(user_id=assistant.user_id, assistant_id=assistant.assistant.id,
                                 topic_threads=threads_po)

    def to_assistant(self) -> OpenAiAssistant:
        assistant = client.beta.assistants.retrieve(self.assistant_id)
        threads = {thread_po.topic: thread_po.thread_id for thread_po in self.topic_threads}
        return OpenAiAssistant(user_id=self.user_id, assistant=assistant, threads=threads)


class OpenAiAssistantDAO(metaclass=SingletonMeta):
    _singleton_instance = None

    def __init__(self, filepath: str):
        # This ensures that initialization happens only once
        self.assistants = {}
        if not OpenAiAssistantDAO._singleton_instance:
            self.filepath = filepath or PROJECT_ROOT / 'data/agents.q'
            self._lock = Lock()
            self._load_assistants()
            OpenAiAssistantDAO._singleton_instance = self

    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT / 'data/agents.q') -> "OpenAiAssistantDAO":
        """
        The factory method for getting the singleton instance.
        """
        if not cls._singleton_instance:
            cls(filepath)
        return cls._singleton_instance

    def _load_assistants(self):
        try:
            with open(self.filepath, 'r') as file:
                self.assistants = json.load(file)
        except FileNotFoundError:
            self.assistants = {}

    def _save(self):
        with open(self.filepath, 'w') as file:
            json.dump(self.assistants, file, ensure_ascii=False, indent=4)

    def add_assistant(self, assistant_po: OpenAiAssistantPO):
        if assistant_po:
            with self._lock:
                self.assistants[assistant_po.assistant_id] = assistant_po.model_dump()
                self._save()

    def get_assistant(self, assistant_id) -> Optional[OpenAiAssistantPO]:
        with self._lock:
            assistant_dict = self.assistants.get(assistant_id)
            if assistant_dict:
                return OpenAiAssistantPO(**assistant_dict)
        return None

    def update_assistant(self, updated_assistant_po: OpenAiAssistantPO):
        with self._lock:
            assistant_dict = self.assistants.get(updated_assistant_po.assistant_id)
            if assistant_dict:
                self.assistants[updated_assistant_po.assistant_id] = updated_assistant_po.model_dump()
                self._save()
                return
            else:
                raise ValueError(f"assistant with assistant_id {updated_assistant_po.assistant_id} not found")

    def delete_assistant(self, assistant_id):
        with self._lock:
            self.assistants.pop(assistant_id)
            self._save()

    def get_my_assistant_ids(self, user_id) -> List[str]:
        with self._lock:
            result: List[str] = []
            for assistant_dict in self.assistants.values():
                if assistant_dict.get('user_id') == user_id:
                    result.append(assistant_dict['assistant_id'])
            return result
