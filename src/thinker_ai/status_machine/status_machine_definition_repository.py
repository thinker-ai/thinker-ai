import json
import os
from json import JSONDecodeError
from typing import Dict, Any, Set
from thinker_ai.status_machine.state_machine_definition import (StateMachineDefinition,
                                                                StateMachineDefinitionRepository,
                                                                StateMachineDefinitionBuilder)


class DefaultBasedStateMachineDefinitionRepository(StateMachineDefinitionRepository):
    state_machine_definition_builder = StateMachineDefinitionBuilder()

    def __init__(self, definitions: Dict[str, Dict[str, Any]] = None):
        if not definitions:
            self.definitions = {}
        else:
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
            except JSONDecodeError:
                definitions = {}
        return cls(definitions)

    def get_root(self, state_machine_def_group_name: str) -> StateMachineDefinition:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            return self.state_machine_definition_builder.root_from_dict(state_machine_def_group_name,
                                                                        state_machine_def_group)

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def get_root_state_machine_name(self, state_machine_def_group_name: str) -> str:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            for name, definition in state_machine_def_group.items():
                if definition.get("is_root"):
                    return name

    def get_state_machine_names(self, state_machine_def_group_name: str) -> Set:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            return set(state_machine_def_group.keys())

    def get(self, state_machine_def_group_name: str, state_machine_name: str) -> StateMachineDefinition:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            data = state_machine_def_group.get(state_machine_name)
            if data:
                return (self.state_machine_definition_builder.
                        state_machine_def_from_dict(group_name=state_machine_def_group_name,
                                                    state_machine_name=state_machine_name,
                                                    data=data,
                                                    exist_state_machine_names=self.get_state_machine_names(
                                                        state_machine_def_group_name)))

    def set(self, state_machine_def_group_name: str, state_machine_def: StateMachineDefinition):
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if not state_machine_def_group:
            state_machine_def_group = {}
            state_machine_def.is_root = True
            state_machine_def_group[
                state_machine_def.name] = self.state_machine_definition_builder.state_machine_def_to_dict(
                state_machine_def)
        else:
            state_machine_def_group[
                state_machine_def.name] = self.state_machine_definition_builder.state_machine_def_to_dict(
                state_machine_def)
