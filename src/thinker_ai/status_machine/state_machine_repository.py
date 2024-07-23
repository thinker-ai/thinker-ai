import json
import os
from typing import Dict, Any

from thinker_ai.status_machine.state_machine import StateMachineContext, StateMachineContextRepository, \
    StateMachineDefinitionRepository, StateContextBuilder


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
        return self._state_machine_from_dict(id, data)

    @staticmethod
    def _state_machine_to_dict(state_machine: StateMachineContext) -> Dict[str, Any]:
        current_context_data = {
            "id": state_machine.current_state_context.id,
            "state_def_id": state_machine.current_state_context.state_def.id,
        }

        history_data = [
            {
                "id": context.id,
                "state_def_id": context.state_def.id,
            }
            for context in state_machine.history
        ]

        return {
            "definition_id": state_machine.state_machine_def_id,
            "current_context": current_context_data,
            "history": history_data
        }

    def _state_machine_from_dict(self, id: str, data: Dict[str, Any]) -> StateMachineContext:
        definition = self.state_machine_definition_repository.load(data["definition_id"])

        current_context_data = data["current_context"]
        current_state = next(
            sd for sd in definition.states_def if sd.id == current_context_data["state_def_id"]
        )
        current_state_context = self.state_context_builder.build(current_state,
                                                                 current_context_data["id"])

        history = [self.state_context_builder.build(
            next(sd for sd in definition.states_def if sd.id == context_data["state_def_id"]),
            context_data["id"])
            for context_data in data["history"]]

        state_machine_context = StateMachineContext(
            id=id,
            definition_id=data["definition_id"],
            current_context=current_state_context,
            state_context_builder=self.state_context_builder,
            history=history,
            state_machine_context_repository=self,
            state_machine_definition_repository=self.state_machine_definition_repository
        )
        return state_machine_context
