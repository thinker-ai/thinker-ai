import json
import os
from json import JSONDecodeError
from typing import Dict, Any, Set, Optional
from thinker_ai.status_machine.state_machine_definition import (StateMachineDefinition,
                                                                StateMachineDefinitionRepository,
                                                                StateMachineDefinitionBuilder)


class DefaultBasedStateMachineDefinitionRepository(StateMachineDefinitionRepository):

    def __init__(self, definitions: Dict[str, Dict[str, Any]] = None, base_dir: str = None, file_name: str = None):
        self.base_dir = base_dir
        self.file_name = file_name
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

    def group_to_json(self, state_machine_def_group_name: str) -> str:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            return json.dumps(state_machine_def_group, indent=2, ensure_ascii=False)
        else:
            return ""

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
            return StateMachineDefinitionBuilder.root_from_group_def_dict(state_machine_def_group_name,
                                                                          state_machine_def_group,self)

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def save(self):
        if self.base_dir and self.file_name:
            self.to_file(self.base_dir, self.file_name)
        else:
            raise Exception('No base dir or file_name specified')

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
        return set()

    def get(self, state_machine_def_group_name: str, state_machine_name: str) -> Optional[StateMachineDefinition]:
        state_machine_def_group = self.definitions.get(state_machine_def_group_name)
        if state_machine_def_group:
            data = state_machine_def_group.get(state_machine_name)
            if data:
                return StateMachineDefinitionBuilder.from_dict(
                    group_def_name=state_machine_def_group_name,
                    state_machine_def_name=state_machine_name,
                    state_machine_def_dict=data,
                    state_machine_definition_repository=self
                )
        return None

    def set_def(self, state_machine_def_group_name: str, state_machine_def: StateMachineDefinition):
        state_machine_def_dict = StateMachineDefinitionBuilder.to_dict(state_machine_def)
        if state_machine_def.is_root:
            state_machine_def_dict[list(state_machine_def_dict.keys())[0]]['is_root'] = True
        self.set_dict(state_machine_def_group_name,state_machine_def_dict)

    def set_dict(self, state_machine_def_group_name: str, state_machine_def_dict: dict):
        state_machine_def_group_dict = self.definitions.get(state_machine_def_group_name)
        if not state_machine_def_group_dict:
            state_machine_def_group_dict = {}
            self.definitions[state_machine_def_group_name] = state_machine_def_group_dict
            state_machine_def_dict["is_root"] = True
        state_machine_def_group_dict.update(state_machine_def_dict)
