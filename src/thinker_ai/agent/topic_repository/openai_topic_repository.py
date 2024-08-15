import json
from json import JSONDecodeError
from threading import Lock
from typing import Optional, List

from thinker_ai.agent.topic_repository.topic_builder import TopicInfo
from thinker_ai.agent.topic_repository.topic_repository import TopicInfoRepository
from thinker_ai.configs.const import PROJECT_ROOT


class OpenAiTopicInfoRepository(TopicInfoRepository):
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT / 'data/topics.json') -> "OpenAiTopicInfoRepository":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(filepath)
        return cls._instance

    def __init__(self, filepath: str):
        if not OpenAiTopicInfoRepository._instance:
            self.filepath = filepath
            self._data = self._load_data()
            OpenAiTopicInfoRepository._instance = self
        else:
            raise Exception("Attempting to instantiate a singleton class.")

    def _load_data(self):
        try:
            with open(self.filepath, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, JSONDecodeError):
            return {}

    def _save_data(self):
        with open(self.filepath, 'w') as file:
            json.dump(self._data, file, indent=4)

    @staticmethod
    def _to_key(user_id: str, topic_name: str):
        key = f"{user_id}_{topic_name}"
        return key

    def add(self, topic_info: TopicInfo):
        key = self._to_key(topic_info.user_id, topic_info.topic.name)
        self._data[key] = topic_info.to_dict()
        self._save_data()

    def update(self, topic_info: TopicInfo):
        key = self._to_key(topic_info.user_id, topic_info.topic.name)
        if key in self._data:
            self._data[key] = topic_info.to_dict()
            self._save_data()

    def delete(self, user_id: str, topic_name: str):
        key = self._to_key(user_id, topic_name)
        del self._data[key]
        self._save_data()

    def get_by(self, user_id: str, topic_name: str) -> Optional[TopicInfo]:
        key = self._to_key(user_id, topic_name)
        data_dict = self._data.get(key)
        if data_dict:
            return TopicInfo.from_dict(data_dict)

    def get_all(self, user_id: str) -> List[TopicInfo]:
        return [TopicInfo.from_dict(v) for v in self._data.values() if v.user_id == user_id]

    @classmethod
    def reset_instance(cls):
        cls._instance = None
