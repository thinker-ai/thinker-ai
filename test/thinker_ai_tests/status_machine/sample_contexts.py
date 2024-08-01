from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository, StateDefinition, BaseStateDefinition
from thinker_ai.status_machine.state_machine_instance import BaseStateContext, StateMachineRepository, StateContext, \
    CompositeStateContext, DefaultStateContextBuilder


class EndSampleContext(BaseStateContext):
    def __init__(self, id: str, state_def: BaseStateDefinition):
        super().__init__(id, state_def)


class SampleContext(StateContext):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)


class CompositeSampleContext(CompositeStateContext):
    def __init__(self, id: str, state_def: StateDefinition,
                 state_context_builder_class: DefaultStateContextBuilder, state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, state_def, state_context_builder_class, state_machine_repository,
                         state_machine_definition_repository)
