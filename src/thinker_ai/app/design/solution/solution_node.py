from typing import Optional, Type, Set

from thinker_ai.status_machine.state_machine_definition import StateMachineDefinition, StateMachineDefinitionBuilder


class SolutionResult:
    def __init__(self, result: str, is_success: bool, code: Optional[str] = None):
        self.result = result
        self.is_success = is_success
        self.code = code


class PlanResult:
    def __init__(self, is_success: bool, state_machine_definition: Optional[StateMachineDefinition] = None,
                 message: Optional[str] = None):
        self.state_machine_definition = state_machine_definition
        self.is_success = is_success
        self.message: Optional[str] = message
