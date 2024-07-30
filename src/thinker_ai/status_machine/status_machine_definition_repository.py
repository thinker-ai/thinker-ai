import json
import os
from typing import Dict, Any, Set
from thinker_ai.status_machine.state_machine import (StateMachineDefinition,
                                                     StateMachineDefinitionRepository,
                                                     StateMachineDefinitionBuilder, BaseStateDefinition,
                                                     StateDefinition)


class DefaultBasedStateMachineDefinitionRepository(StateMachineDefinitionRepository):
    state_machine_definition_builder = StateMachineDefinitionBuilder()

    def __init__(self, definitions: Dict[str, Any]):
        self.definitions = definitions

    @classmethod
    def from_json(cls, json_text) -> "DefaultBasedStateMachineDefinitionRepository":
        definitions = json.loads(json_text)
        return cls(definitions)

    def to_json(self) -> str:
        return json.dumps(self.definitions, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, base_dir: str, file_name: str) -> "DefaultBasedStateMachineDefinitionRepository":
        file_path = os.path.join(base_dir, file_name)
        definitions = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    definitions = json.load(file)
            except Exception:
                pass
        return cls(definitions)

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def get_root(self) -> StateMachineDefinition:
        return self.state_machine_definition_builder.root_from_dict(self.definitions)

    def get_root_name(self) -> str:
        for name, definition in self.definitions.items():
            if definition.get("is_root"):
                return name

    def get_state_machine_names(self) -> Set:
        return set(self.definitions.keys())

    def get(self, name: str) -> StateMachineDefinition:
        data = self.definitions.get(name)
        if data:
            return self.state_machine_definition_builder.state_machine_def_from_dict(name, data,
                                                                                     self.get_state_machine_names())

    def set(self, name: str, state_machine_def: StateMachineDefinition):
        states_def = self.get_root().get_state_execute_paths()
        if not states_def:
            state_machine_def.is_root = True
            self.definitions[name] = self.state_machine_definition_builder.state_machine_def_to_dict(state_machine_def)
        else:
            for state_def_name, state_def in states_def:
                if state_def_name == name and isinstance(state_def, StateDefinition):
                    state_def.is_composite = True
                    self.definitions[name] = self.state_machine_definition_builder.state_machine_def_to_dict(
                        state_machine_def)