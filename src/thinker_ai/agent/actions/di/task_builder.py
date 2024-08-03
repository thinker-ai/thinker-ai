import uuid
from typing import Optional

from thinker_ai.agent.actions.di.state_machine_task import StateMachineTask
from thinker_ai.agent.actions.di.state_task import StateTask
from thinker_ai.status_machine.state_machine_definition import StateDefinition, BaseStateDefinition, StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_context import StateContextBuilder, StateContext, BaseStateContext,StateMachineRepository


class TaskBuilder(StateContextBuilder):
    @staticmethod
    def build_state_instance(state_def: StateDefinition,
                             state_context_class_name: str,
                             instance_id: Optional[str] = str(uuid.uuid4())) -> StateContext:
        return StateTask(instance_id=instance_id, state_def=state_def)

    @staticmethod
    def build_terminal_state_instance(state_def: BaseStateDefinition,
                                      state_context_class_name: str,
                                      instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        return BaseStateContext(instance_id=instance_id, state_def=state_def)

    @classmethod
    def build_composite_state_instance(cls, state_def: StateDefinition,
                                       state_context_class_name: str,
                                       state_machine_definition_repository: StateMachineDefinitionRepository,
                                       state_machine_context_repository: StateMachineRepository,
                                       instance_group_id: str,
                                       instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        if state_def.is_composite:
            return StateMachineTask(instance_id=instance_id,
                                    instance_group_id=instance_group_id,
                                    state_def=state_def,
                                    state_context_builder_class=cls,
                                    state_machine_context_repository=state_machine_context_repository,
                                    state_machine_definition_repository=state_machine_definition_repository
                                    )
