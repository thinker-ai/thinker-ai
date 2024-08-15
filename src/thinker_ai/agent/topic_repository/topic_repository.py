from abc import ABC, abstractmethod
from typing import Optional

from thinker_ai.agent.topic_repository.topic_builder import TopicInfo


class TopicInfoRepository(ABC):
    @abstractmethod
    def add(self, topic_info: TopicInfo):
        raise NotImplementedError

    @abstractmethod
    def update(self, topic_info: TopicInfo):
        raise NotImplementedError

    @abstractmethod
    def delete(self, user_id: str, topic_name: str):
        raise NotImplementedError

    @abstractmethod
    def get_by(self, user_id: str, topic_name: str) -> Optional[TopicInfo]:
        raise NotImplementedError

    @abstractmethod
    def get_all(self, user_id: str) -> list[TopicInfo]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reset_instance(cls):
        raise NotImplementedError
