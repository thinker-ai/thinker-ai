from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository, StateDefinition, BaseStateDefinition
from thinker_ai.status_machine.state_machine_scenario import BaseStateScenario, StateMachineRepository, StateScenario, \
    CompositeStateScenario, DefaultStateContextBuilder


class EndSampleContext(BaseStateScenario):
    def __init__(self, id: str, state_def: BaseStateDefinition):
        super().__init__(id, state_def)


class SampleContext(StateScenario):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)


class CompositeSampleContext(CompositeStateScenario):
    def __init__(self, id: str, state_def: StateDefinition,
                 state_scenario_builder_class: DefaultStateContextBuilder, state_machine_scenario_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, state_def, state_scenario_builder_class, state_machine_scenario_repository,
                         state_machine_definition_repository)
