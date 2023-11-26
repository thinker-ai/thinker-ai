from thinker_ai.agent.action_message import ActionMessage
from thinker_ai.memory.longterm_memory import LongTermMemory


class RoleLongTermMemory(LongTermMemory):

    def __init__(self, rc: "RoleContext"):
        super().__init__(f'{rc.roles_folder}/{rc.role_id}.json')
        self._rc = rc

    def add(self, message: ActionMessage):
        for action in self._rc.watch:
            if message.cause_by == action:
                super().add(message)

