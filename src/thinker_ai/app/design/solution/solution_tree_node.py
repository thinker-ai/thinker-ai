from typing import Optional, Type

from thinker_ai.app.design.solution.solution_node_builder import SolutionNodeBuilder
from thinker_ai.status_machine.state_machine_definition import StateDefinition, StateMachineDefinition, \
    StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import CompositeStateScenario, StateMachineScenarioRepository


class PlanResult:
    def __init__(self, is_success: bool, state_machine_definition: Optional[StateMachineDefinition] = None,message: Optional[str] = None):
        self.state_machine_definition = state_machine_definition
        self.is_success = is_success
        self.message: Optional[str] = message


class SolutionTreeNode(CompositeStateScenario):
    def __init__(self,
                 scenario_id: str,
                 scenario_group_id: str,
                 state_def: StateDefinition,
                 state_scenario_builder_class: Type[SolutionNodeBuilder],
                 state_machine_scenario_repository: StateMachineScenarioRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository
                 ):
        super().__init__(scenario_id, scenario_group_id, state_def, state_scenario_builder_class,
                         state_machine_scenario_repository, state_machine_definition_repository)
