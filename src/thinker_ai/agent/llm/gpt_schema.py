from __future__ import annotations

from dataclasses import dataclass, field
from thinker_ai.utils.logs import logger


@dataclass
class Message:
    """list[<agent>: <content>]"""
    content: str
    role: str = field(default='user')  # system / user / agent / function

    def __str__(self):
        # prefix = '-'.join([self.agent, str(self.cause_by)])
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
        super().__init__(content, 'agent')


class PromptMessage:
    user_message: UserMessage
    system_message: SystemMessage

    def __init__(self, user_msg: str, sys_msg: str = None):
        self.user_message = UserMessage(user_msg)
        self.system_message = SystemMessage(sys_msg or 'You are a helpful agent.')

    @property
    def user_message_content(self) -> str:
        return self.user_message.content

    @property
    def system_message_content(self) -> str:
        return self.system_message.content

    def to_dicts(self) -> list[dict]:
        return [self.user_message.to_dict(), self.system_message.to_dict()]


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)
