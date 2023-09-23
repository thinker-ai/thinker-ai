from pathlib import Path

from thinker_ai.llm.schema import Message
from thinker_ai.memory.longterm_memory import LongTermMemory


class RoleLongTermMemory(LongTermMemory):

    def __init__(self, rc: "RoleContext"):
        super().__init__(f'{rc.roles_folder}/{rc.role_id}.json')
        self._rc = rc

    def add(self, message: Message):
        for action in self._rc.watch:
            if message.cause_by == action:
                super().add(message)

