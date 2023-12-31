from collections import defaultdict
from typing import Iterable, Type

from thinker_ai.actions.action import BaseAction
from thinker_ai.work_flow.tasks import TaskMessage


class Memory:
    """The most basic memory: super-memory"""

    def __init__(self):
        """Initialize an empty storage list and an empty index dictionary"""
        self.storage: list[TaskMessage] = []
        self.index: dict[Type[BaseAction], list[TaskMessage]] = defaultdict(list)

    def add(self, message: TaskMessage):
        """Add a new message to storage, while updating the index"""
        if message in self.storage:
            return
        self.storage.append(message)
        if message.cause_by:
            self.index[message.cause_by].append(message)

    def add_batch(self, messages: Iterable[TaskMessage]):
        for message in messages:
            self.add(message)

    def get_by_role(self, role: str) -> list[TaskMessage]:
        """Return all messages of a specified agent"""
        return [message for message in self.storage if message.role == role]

    def get_by_content(self, content: str) -> list[TaskMessage]:
        """Return all messages containing a specified content"""
        return [message for message in self.storage if content in message.content]

    def delete(self, message: TaskMessage):
        """Delete the specified message from storage, while updating the index"""
        self.storage.remove(message)
        if message.cause_by and message in self.index[message.cause_by]:
            self.index[message.cause_by].remove(message)

    def clear(self):
        """Clear storage and index"""
        self.storage = []
        self.index = defaultdict(list)

    def count(self) -> int:
        """Return the number of messages in storage"""
        return len(self.storage)

    def try_remember(self, keyword: str) -> list[TaskMessage]:
        """Try to recall all messages containing a specified keyword"""
        return [message for message in self.storage if keyword in message.content]

    def get(self, k=0) -> list[TaskMessage]:
        """Return the most recent k memories, return all when k=0"""
        return self.storage[-k:]

    def filter_new_observes(self, observed: list[TaskMessage], k=0) -> list[TaskMessage]:
        """remember the most recent k memories from observed Messages, return all when k=0"""
        already_observed = self.get(k)
        news: list[TaskMessage] = []
        for i in observed:
            if i in already_observed:
                continue
            news.append(i)
        return news

    def get_by_action(self, action: Type[BaseAction]) -> list[TaskMessage]:
        """Return all messages triggered by a specified Action"""
        return self.index[action]

    def get_by_actions(self, actions: Iterable[Type[BaseAction]]) -> list[TaskMessage]:
        """Return all messages triggered by specified Actions"""
        rsp = []
        for action in actions:
            if action not in self.index:
                continue
            rsp += self.index[action]
        return rsp
