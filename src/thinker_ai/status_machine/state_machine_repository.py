import json
import os
from typing import Dict, Any

from thinker_ai.status_machine.state_machine import StateMachineContext, StateMachineContextRepository, \
    StateMachineDefinitionRepository, StateContext, StateContextBuilder, CompositeStateContext


class FileBasedStateMachineContextRepository(StateMachineContextRepository):
    def __init__(self, base_dir: str, file_name: str,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        self.base_dir = base_dir
        self.file_path = os.path.join(base_dir, file_name)
        self.state_machine_definition_repository = state_machine_definition_repository
        self.state_context_builder = StateContextBuilder(self, self.state_machine_definition_repository)
        self.instances = self._load_instances()

    def _load_instances(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        return {}

    def save(self, state_machine_context: StateMachineContext):
        self.instances[state_machine_context.id] = self._state_machine_to_dict(state_machine_context)
        with open(self.file_path, 'w') as file:
            json.dump(self.instances, file, indent=2)

    def load(self, id: str) -> StateMachineContext:
        if id not in self.instances:
            raise ValueError(f"StateMachine instance with id '{id}' not found")

        data = self.instances[id]
        return self._state_machine_from_dict(data)

    @staticmethod
    def _state_machine_to_dict(state_machine: StateMachineContext) -> Dict[str, Any]:
        current_context_data = {
            "id": state_machine.current_context.id,
            "state_id": state_machine.current_context.state.id,
        }

        history_data = [
            {
                "id": context.id,
                "state_id": context.state.id,
            }
            for context in state_machine.history
        ]

        if isinstance(state_machine.current_context, CompositeStateContext):
            current_context_data["inner_state_machine_id"] = state_machine.current_context.inner_state_machine_id

        return {
            "id": state_machine.id,
            "definition_id": state_machine.definition_id,
            "current_context": current_context_data,
            "history": history_data
        }

    def _state_machine_from_dict(self, data: Dict[str, Any]) -> StateMachineContext:
        definition = self.state_machine_definition_repository.load(data["definition_id"])

        current_context_data = data["current_context"]
        current_state = next(
            sd for sd in definition.states if sd.id == current_context_data["state_id"]
        )
        current_state_context = self.state_context_builder.build(current_state,
                                                                 current_context_data["id"],
                                                                 current_context_data.get("inner_state_machine_id"))

        history = [self.state_context_builder.build(
            next(sd for sd in definition.states if sd.id == context_data["state_id"]),
            context_data.get("id"),
            context_data.get("inner_state_machine_id"))
            for context_data in data["history"]]

        state_machine_context = StateMachineContext(
            id=data["id"],
            definition_id=data["definition_id"],
            current_state_context=current_state_context,
            state_context_builder=self.state_context_builder,
            history=history,
            state_machine_context_repository=self,
            state_machine_definition_repository=self.state_machine_definition_repository
        )
        return state_machine_context
