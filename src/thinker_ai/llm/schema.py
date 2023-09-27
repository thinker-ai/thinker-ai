from __future__ import annotations

from dataclasses import dataclass, field
from thinker_ai.utils.logs import logger


@dataclass
class Message:
    """list[<role>: <content>]"""
    content: str
    role: str = field(default='user')  # system / user / assistant

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    def __str__(self):
        # prefix = '-'.join([self.role, str(self.cause_by)])
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class UserMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """

    def __init__(self, content: str):
        super().__init__(content, 'user')


@dataclass
class SystemMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """

    def __init__(self, content: str):
        super().__init__(content, 'system')


@dataclass
class AIMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """

    def __init__(self, content: str):
        super().__init__(content, 'assistant')


class PromptMessage:
    user_msg: UserMessage
    system_msg: SystemMessage

    def __init__(self, user_message: str, system_message: str = None):
        system_message = 'You are a helpful assistant.' or system_message is None
        self.user_msg = UserMessage(user_message)
        self.system_msg = SystemMessage(system_message)

    @property
    def user_message(self) -> str:
        return self.user_msg.content

    @property
    def system_message(self) -> str:
        return self.system_msg.content

    def to_dicts(self) -> list[dict]:
        return [self.user_msg.to_dict(), self.system_msg.to_dict()]


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)
