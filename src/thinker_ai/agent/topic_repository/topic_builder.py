
from thinker_ai.configs.config import config
from thinker_ai.configs.llm_config import LLMType


class Topic:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}

    @classmethod
    def from_dict(cls, data):
        return cls(name=data["name"])


class TopicInfo:
    user_id: str
    topic: Topic

    def __init__(self, user_id: str, topic: Topic):
        self.user_id = user_id
        self.topic = topic

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "topic": self.topic.to_dict()
        }

    @classmethod
    def from_dict(cls, data:dict):
        topic = TopicBuilder.create(data["topic"])
        return cls(user_id=data["user_id"], topic=topic)


class OpenAiTopic(Topic):
    def __init__(self, name: str, thread_id: str, assistant_id: str):
        super().__init__(name)
        self.thread_id = thread_id
        self.assistant_id = assistant_id

    def to_dict(self):
        # 调用父类的 to_dict 方法获取基础字段
        base_dict = super().to_dict()
        # 添加子类特有的字段
        base_dict.update({
            "thread_id": self.thread_id,
            "assistant_id": self.assistant_id
        })
        return base_dict

    @classmethod
    def from_dict(cls, data:dict):
        return cls(
            name=data["name"],
            thread_id=data["thread_id"],
            assistant_id=data["assistant_id"]
        )


class TopicBuilder:
    @staticmethod
    def create(data: dict):
        if config.llm.api_type == LLMType.OPENAI:
            return OpenAiTopic.from_dict(data)
        return TopicInfo.from_dict(data)
