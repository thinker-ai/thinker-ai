from typing import Optional

from thinker_ai.status_machine.state_machine_definition import StateDefinition
from thinker_ai.status_machine.state_machine_scenario import StateScenario


class SolutionResult:
    def __init__(self, result: str, is_success: bool, code: Optional[str] = None):
        self.result = result
        self.is_success = is_success
        self.code = code


class SolutionNode(StateScenario):
    def __init__(self,
                 scenario_id: str,
                 state_def: StateDefinition,
                 title: str,
                 description: str,
                 code_path: str):
        super().__init__(scenario_id, state_def)
        self.title = title
        self.description = description
        self.code_path = code_path
