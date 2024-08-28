import uuid
from typing import Optional, Type

from thinker_ai.status_machine.state_machine_definition import StateDefinition, StateMachineDefinitionRepository, \
    StateMachineDefinition, BaseStateDefinition
from thinker_ai.status_machine.state_machine_scenario import StateScenario, CompositeStateScenario, \
    StateMachineScenarioRepository, BaseStateScenario, StateScenarioBuilder


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
        self.code_path = code_path


class PlanResult:
    def __init__(self, is_success: bool, state_machine_definition: Optional[StateMachineDefinition] = None,
                 message: Optional[str] = None):
        self.state_machine_definition = state_machine_definition
        self.is_success = is_success
        self.message: Optional[str] = message


class SolutionTreeNode(CompositeStateScenario):
    def __init__(self,
                 scenario_id: str,
                 scenario_root_id: str,
                 state_def: StateDefinition,
                 state_scenario_builder_class: Type['SolutionNodeBuilder'],
                 state_machine_scenario_repository: StateMachineScenarioRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository
                 ):
        super().__init__(scenario_id, scenario_root_id, state_def, state_scenario_builder_class,
                         state_machine_scenario_repository, state_machine_definition_repository)

    def to_dict(self) -> dict:
        result = dict()
        result['title'] = self.state_def.name
        result['description'] = self.state_def.description
        result['children'] = self.children_to_dict()
        return result

    def children_to_dict(self) -> dict:
        state_machine = super().get_state_machine()
        states_def = state_machine.get_state_machine_def().states_def
        result = dict()
        for state in states_def:
            state_scenario_des = state_machine.creat_state_scenario_des(state)
            result[state_scenario_des.scenario_id] = state_scenario_des.to_dict()
        return result


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
                                       scenario_root_id: str,
                                       scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        if state_def.is_composite:
            return SolutionTreeNode(scenario_id=scenario_id,
                                    scenario_root_id=scenario_root_id,
                                    state_def=state_def,
                                    state_scenario_builder_class=cls,
                                    state_machine_scenario_repository=state_machine_scenario_repository,
                                    state_machine_definition_repository=state_machine_definition_repository)
