
from thinker_ai.agent.actions import Action
from thinker_ai.agent.provider.schema import Message


class ExecuteTask(Action):
    name: str = "ExecuteTask"
    i_context: list[Message] = []

    async def run(self, *args, **kwargs):
        pass
