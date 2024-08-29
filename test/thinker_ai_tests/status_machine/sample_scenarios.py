from thinker_ai.status_machine.state_machine_definition import StateMachineDefinition, StateDefinition, \
    BaseStateDefinition, CompositeStateDefinition
from thinker_ai.status_machine.state_machine_scenario import BaseStateScenario, StateMachineScenarioRepository, \
    StateScenario, CompositeStateScenario, DefaultStateScenarioBuilder
from typing import Type


class EndSampleScenario(BaseStateScenario):
    def __init__(self, id: str, state_def: BaseStateDefinition):
        super().__init__(id, state_def)


class SampleScenario(StateScenario):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)


class CompositeSampleScenario(CompositeStateScenario):
    def __init__(self,
                 id: str,
                 root_id: str,
                 state_def: CompositeStateDefinition,
                 state_scenario_builder_class: Type[DefaultStateScenarioBuilder],
                 state_machine_scenario_repository: StateMachineScenarioRepository):
        super().__init__(id,
                         root_id,
                         state_def,
                         state_scenario_builder_class,
                         state_machine_scenario_repository)
