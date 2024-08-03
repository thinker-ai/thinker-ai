from typing import Optional

from thinker_ai.status_machine.state_machine_definition import StateDefinition
from thinker_ai.status_machine.state_machine_scenario import StateScenario


class StateTaskResult:
    def __init__(self, result: str, is_success: bool, code: Optional[str] = None):
        self.result = result
        self.is_success = is_success
        self.code = code


class StateTask(StateScenario):
    def __init__(self, scenario_id: str, state_def: StateDefinition):
        super().__init__(scenario_id, state_def)
