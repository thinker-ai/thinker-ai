from thinker_ai.status_machine.state_machine_definition import StateDefinition
from thinker_ai.status_machine.state_machine_context import StateContext


class StateTask(StateContext):
    def __init__(self, instance_id: str, state_def: StateDefinition):
        super().__init__(instance_id, state_def)
