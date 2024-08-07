import json
from threading import Lock
from typing import List, Optional, Dict, Literal

from thinker_ai.agent.openai_assistant_api import openai_client
from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.common.serializable import Serializable
from thinker_ai.common.singleton_meta import SingletonMeta


class OpenAiAssistantApiPO(Serializable):
    user_id: str
    name: str
    assistant_id: str
    topic_threads: Dict[str, str]

    @classmethod
    def from_assistant_api(cls, assistant_api: OpenAiAssistantApi) -> "OpenAiAssistantApiPO":
        return OpenAiAssistantApiPO(user_id=assistant_api.user_id, assistant_id=assistant_api.assistant.id,
                                    name=assistant_api.name, topic_threads=assistant_api.topic_threads)

    def to_assistant_api(self) -> OpenAiAssistantApi:
        assistant = openai_client.beta.assistants.retrieve(self.assistant_id)
        return OpenAiAssistantApi(user_id=self.user_id, assistant=assistant, topic_threads=self.topic_threads)


class OpenAiAssistantApiDAO(metaclass=SingletonMeta):
    _singleton_instance = None

    def __init__(self, filepath: str):
        # This ensures that initialization happens only once
        self.assistant_apis = {}
        if not OpenAiAssistantApiDAO._singleton_instance:
            self.filepath = filepath or PROJECT_ROOT / 'data/test_assistants.json'
            self._lock = Lock()
            self._load_assistant_apis()
            OpenAiAssistantApiDAO._singleton_instance = self

    @classmethod
    def get_instance(cls,
                     filepath: Optional[str] = PROJECT_ROOT / 'data/test_assistants.json') -> "OpenAiAssistantApiDAO":
        """
        The factory method for getting the singleton instance.
        """
        if not cls._singleton_instance:
            cls(filepath)
        return cls._singleton_instance

    def _load_assistant_apis(self):
        try:
            with open(self.filepath, 'r') as file:
                self.assistant_apis = json.load(file)
        except FileNotFoundError:
            self.assistant_apis = {}

    def _save(self):
        with open(self.filepath, 'w') as file:
            json.dump(self.assistant_apis, file, ensure_ascii=False, indent=4)

    def add_assistant_api(self, assistant_api_po: OpenAiAssistantApiPO):
        if assistant_api_po:
            with self._lock:
                self.assistant_apis[assistant_api_po.assistant_id] = assistant_api_po.model_dump()
                self._save()

    def get_assistant_api_by_id(self, assistant_id: str) -> Optional[OpenAiAssistantApiPO]:
        with self._lock:
            assistant_api_dict = self.assistant_apis.get(assistant_id)
            if assistant_api_dict:
                return OpenAiAssistantApiPO(**assistant_api_dict)
        return None

    # 不同用户的assistant name可以相同
    def get_assistant_api_by_name(self, user_id: str, name: str) -> Optional[OpenAiAssistantApiPO]:
        with self._lock:
            for assistant_api_dict in self.assistant_apis.values():
                if assistant_api_dict.get("name") == name and assistant_api_dict.get("user_id") == user_id:
                    return OpenAiAssistantApiPO(**assistant_api_dict)
        return None

    def update(self, assistant_api_po: OpenAiAssistantApiPO):
        with self._lock:
            assistant_api_dict = self.assistant_apis.get(assistant_api_po.assistant_id)
            if assistant_api_dict:
                self.assistant_apis[assistant_api_po.assistant_id] = assistant_api_po.model_dump()
                self._save()
            else:
                raise ValueError(
                    f"assistant_api with assistant_id {assistant_api_po.assistant_id} not found")

    def delete_assistant_api(self, assistant_id):
        with self._lock:
            self.assistant_apis.pop(assistant_id)
            self._save()

    def get_my_assistants_of(self, user_id, of: Literal["assistant_id", "name"]) -> List[str]:
        with self._lock:
            result: List[str] = []
            for assistant_api_dict in self.assistant_apis.values():
                if assistant_api_dict["user_id"] == user_id:
                    result.append(assistant_api_dict.get(of))
            return result

    def add_topic(self, user_id, assistant_id, topic_name, thread_id):
        with self._lock:
            assistant_api_dict = self.assistant_apis.get(assistant_id)
            if assistant_api_dict:
                if assistant_api_dict["user_id"] != user_id:
                    raise PermissionError("Cannot update an topic for another user.")
                topic_threads = assistant_api_dict["topic_threads"]
                topic_threads[topic_name] = thread_id
                self._save()

    def del_topic(self, user_id: str, assistant_id: str, topic_name: str) -> Optional[str]:
        with self._lock:
            assistant_api_dict = self.assistant_apis.get(assistant_id)
            if assistant_api_dict:
                if assistant_api_dict["user_id"] != user_id:
                    raise PermissionError("Cannot update a topic for another user.")
                topic_threads = assistant_api_dict.get("topic_threads", {})
                thread_id = topic_threads.get(topic_name)
                if thread_id is not None:
                    del topic_threads[topic_name]
                    self._save()
                    return thread_id
                else:
                    return None  # Topic name does not exist
            else:
                return None  # Assistant ID does not exist

