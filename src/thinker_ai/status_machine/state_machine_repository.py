import json
import os
from typing import Dict, Any

from thinker_ai.status_machine.state_machine import StateMachine, StateMachineRepository, \
    StateMachineDefinitionRepository, StateContextBuilder


class FileBasedStateMachineContextRepository(StateMachineRepository):
    def __init__(self, instances: dict, state_machine_definition_repository: StateMachineDefinitionRepository):
        self.state_machine_definition_repository = state_machine_definition_repository
        self.state_context_builder = StateContextBuilder(self, self.state_machine_definition_repository)
        self.instances = instances

    @classmethod
    def from_file(cls, base_dir: str, file_name: str,
                  state_machine_definition_repository: StateMachineDefinitionRepository) -> "FileBasedStateMachineContextRepository":
        file_path = os.path.join(base_dir, file_name)
        instances = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    instances = json.load(file)
            except Exception:
                pass
        return cls(instances, state_machine_definition_repository)

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.instances, file, indent=2)

    @classmethod
    def form_json(cls, json_text,
                  state_machine_definition_repository: StateMachineDefinitionRepository) -> "FileBasedStateMachineContextRepository":
        instances: dict = json.loads(json_text)
        return cls(instances, state_machine_definition_repository)

    def to_json(self) -> str:
        return json.dumps(self.instances, indent=2, ensure_ascii=False)

    def set(self, id, state_machine_context: StateMachine):
        self.instances[id] = self._state_machine_to_dict(state_machine_context)

    def get(self, id: str) -> StateMachine:
        data = self.instances.get(id)
        if data:
            return self._state_machine_from_dict(id, data)

    @staticmethod
    def _state_machine_to_dict(state_machine: StateMachine) -> Dict[str, Any]:
        current_state_context = {
            "id": state_machine.current_state_context.id,
            "state_def_name": state_machine.current_state_context.state_def.name,
        }

        history_data = [
            {
                "id": context.id,
                "state_def_name": context.state_def.name,
            }
            for context in state_machine.history
        ]

        return {
            "state_machine_def_name": state_machine.state_machine_def_name,
            "current_state_context": current_state_context,
            "history": history_data
        }

    def _state_machine_from_dict(self, id: str, data: Dict[str, Any]) -> StateMachine:
        state_machine_def = self.state_machine_definition_repository.get(data["state_machine_def_name"])

        current_state_context = data["current_state_context"]
        current_state = next(
            sd for sd in state_machine_def.states_def if sd.name == current_state_context["state_def_name"]
        )
        current_state_context = self.state_context_builder.build(state_def=current_state,
                                                                 id=current_state_context["id"])

        history = [self.state_context_builder.build(
            state_def=next(sd for sd in state_machine_def.states_def if sd.name == context_data["state_def_name"]),
            id=context_data["id"])
            for context_data in data["history"]]

        state_machine = StateMachine(
            id=id,
            state_machine_def_name=data["state_machine_def_name"],
            current_state_context=current_state_context,
            state_context_builder=self.state_context_builder,
            history=history,
            state_machine_repository=self,
            state_machine_definition_repository=self.state_machine_definition_repository
        )
        return state_machine
