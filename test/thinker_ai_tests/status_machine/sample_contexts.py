from thinker_ai.status_machine.state_machine import StateContext, CompositeStateContext, CompositeStateDefinition, \
    StateContextBuilder, StateMachineDefinitionRepository, StateMachineRepository, EndStateContext, StateDefinition, \
    EndStateDefinition


class EndSampleContext(EndStateContext):
    def __init__(self, id: str, state_def: EndStateDefinition):
        super().__init__(id, state_def)


class SampleContext(StateContext):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)


class CompositeSampleContext(CompositeStateContext):
    def __init__(self, id: str, composite_state_def: CompositeStateDefinition,
                 state_context_builder: StateContextBuilder, state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository, *args, **kwargs):
        super().__init__(id, composite_state_def, state_context_builder, state_machine_repository,
                         state_machine_definition_repository)
