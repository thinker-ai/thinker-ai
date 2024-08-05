from typing import Optional

from thinker_ai.agent.actions.state_machine_task.task_builder import TaskBuilder
from thinker_ai.agent.actions.state_machine_task.task_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.status_machine.state_machine_definition import StateDefinition, StateMachineDefinition
from thinker_ai.status_machine.state_machine_scenario import CompositeStateScenario


class PlanResult:
    def __init__(self, is_success: bool, state_machine_definition: Optional[StateMachineDefinition] = None,message: Optional[str] = None):
        self.state_machine_definition = state_machine_definition
        self.is_success = is_success
        self.message: Optional[str] = message


class CompositeTask(CompositeStateScenario):
    def __init__(self,
                 scenario_id: str,
                 scenario_group_id: str,
                 state_def: StateDefinition,
                 ):
        super().__init__(scenario_id, scenario_group_id, state_def, TaskBuilder,
                         state_machine_scenario_repository, state_machine_definition_repository)
