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
    _user_message: UserMessage
    _system_message: SystemMessage

    def __init__(self, user_msg: str, sys_msg: str = None):
        system_message = 'You are a helpful assistant.' or sys_msg is None
        self._user_message = UserMessage(user_msg)
        self._system_message = SystemMessage(sys_msg)

    @property
    def user_msg(self) -> str:
        return self._user_message.content

    @property
    def sys_msg(self) -> str:
        return self._system_message.content

    def to_dicts(self) -> list[dict]:
        return [self._user_message.to_dict(), self._system_message.to_dict()]


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)
