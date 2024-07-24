from thinker_ai.status_machine.state_machine import StateContext, CompositeStateContext, CompositeStateDefinition, \
    StateContextBuilder, StateMachineDefinitionRepository, StateMachineRepository


class SampleContext(CompositeStateContext):
    def __init__(self, id: str, composite_state_def: CompositeStateDefinition,
                 state_context_builder: StateContextBuilder, state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository, *args, **kwargs):
        super().__init__(id, composite_state_def, state_context_builder, state_machine_repository,
                         state_machine_definition_repository)


class CompositeSampleContext(CompositeStateContext):
    def __init__(self, id: str, composite_state_def: CompositeStateDefinition,
                 state_context_builder: StateContextBuilder, state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository, *args, **kwargs):
        super().__init__(id, composite_state_def, state_context_builder, state_machine_repository,
                         state_machine_definition_repository)
