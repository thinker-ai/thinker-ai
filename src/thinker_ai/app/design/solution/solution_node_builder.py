import uuid
from typing import Optional

from thinker_ai.app.design.solution.solution_node import SolutionNode
from thinker_ai.app.design.solution.solution_tree_node import SolutionTreeNode
from thinker_ai.status_machine.state_machine_definition import StateDefinition, BaseStateDefinition, \
    StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import StateScenarioBuilder, BaseStateScenario, \
    StateMachineScenarioRepository, StateScenario


class SolutionNodeBuilder(StateScenarioBuilder):
    @staticmethod
    def build_state_scenario(state_def: StateDefinition,
                             state_scenario_class_name: str,
                             scenario_id: Optional[str] = str(uuid.uuid4())) -> StateScenario:
        return SolutionNode(scenario_id=scenario_id, state_def=state_def)

    @staticmethod
    def build_terminal_state_scenario(state_def: BaseStateDefinition,
                                      state_scenario_class_name: str,
                                      scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        return BaseStateScenario(scenario_id=scenario_id, state_def=state_def)

    @classmethod
    def build_composite_state_scenario(cls, state_def: StateDefinition,
                                       state_scenario_class_name: str,
                                       state_machine_definition_repository: StateMachineDefinitionRepository,
                                       state_machine_scenario_repository: StateMachineScenarioRepository,
                                       scenario_group_id: str,
                                       scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        if state_def.is_composite:
            return SolutionTreeNode(scenario_id=scenario_id,
                                    scenario_group_id=scenario_group_id,
                                    state_def=state_def,
                                    state_scenario_builder_class=cls,
                                    state_machine_scenario_repository=state_machine_scenario_repository,
                                    state_machine_definition_repository=state_machine_definition_repository
                                    )
