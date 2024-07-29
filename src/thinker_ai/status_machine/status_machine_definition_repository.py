import json
import os
from typing import Dict, Any, Set, Optional, List, Tuple

from thinker_ai.status_machine.task_desc import TaskType
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
        states_def = self.get_state_execute_order(self.get_root_name())
        if not states_def:
            state_machine_def.is_root = True
            self.definitions[name] = self.state_machine_definition_builder.state_machine_def_to_dict(state_machine_def)
        else:
            for state_def_name, state_def in states_def:
                if state_def_name == name and isinstance(state_def, StateDefinition):
                    state_def.is_composite = True
                    self.definitions[name] = self.state_machine_definition_builder.state_machine_def_to_dict(
                        state_machine_def)

    def get_state_execute_order(self, state_machine_name: str,
                                sorted_state_defs: Optional[List[Tuple[str, BaseStateDefinition]]] = None) \
            -> List[Tuple[str, BaseStateDefinition]]:
        state_machine_def = self.get(state_machine_name)
        if sorted_state_defs is None:
            sorted_state_defs = []
        visited: Set[str] = set()
        stack: List[Tuple[str, BaseStateDefinition]] = []

        if not state_machine_def:
            return sorted_state_defs

        def visit(state: BaseStateDefinition):
            state_def_name = f"{state_machine_def.name}.{state.name}"
            if state_def_name in visited:
                return
            stack.append((state_def_name, state))
            visited.add(state_def_name)
            for transition in state_machine_def.transitions:
                if transition.source.name == state.name:
                    visit(transition.target)

            # Process sub-state machine if any
            if isinstance(state, StateDefinition) and state.is_composite:
                self.get_state_execute_order(f"{state_machine_name}.{state.name}", sorted_state_defs)

        start_state = state_machine_def.get_start_state_def()
        if start_state:
            visit(start_state)

        sorted_state_defs.extend(stack)  # Reverse the stack to get the correct order

        return sorted_state_defs

    def next_state_machine_to_create(self) -> tuple[str, StateDefinition]:
        root_name = self.get_root_name()
        if root_name:
            sorted_states_defs: List[Tuple[str, BaseStateDefinition]] = self.get_state_execute_order(root_name)
            for name, states_def in sorted_states_defs:
                if (name not in self.get_state_machine_names()
                        and isinstance(states_def, StateDefinition)
                        and states_def.task_type.name == TaskType.STATE_MACHINE_PLAN.type_name):
                    return name, states_def
