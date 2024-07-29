import json
import os
from typing import Dict, Any, Set, Optional, List, Tuple

from thinker_ai.status_machine.task_desc import TaskType
from thinker_ai.status_machine.state_machine import (StateMachineDefinition,
                                                     StateMachineDefinitionRepository,
                                                     StateMachineDefinitionBuilder, BaseStateDefinition,
                                                     StateDefinition)


class FileBasedStateMachineDefinitionRepository(StateMachineDefinitionRepository):
    def __init__(self, base_dir: str, file_name: str):
        self.base_dir = base_dir
        self.file_path = os.path.join(base_dir, file_name)
        self.definitions = self._load_definitions()

    def _load_definitions(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as file:
                    return json.load(file)
            except Exception:
                return {}
        return {}

    def load_json_text(self) -> str:
        return json.dumps(self.definitions, indent=2, ensure_ascii=False)

    def build(self, json_text) -> StateMachineDefinition:
        return StateMachineDefinitionBuilder.from_json(json_text, self.get_state_machine_names())

    def save(self, state_machine_def: StateMachineDefinition):
        self.definitions[state_machine_def.name] = StateMachineDefinitionBuilder.state_machine_def_to_dict(
            state_machine_def)
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def get_state_machine_names(self) -> Set:
        return set(self.definitions.keys())

    def load(self, name: str) -> StateMachineDefinition:
        data = self.definitions.get(name)
        if data:
            return StateMachineDefinitionBuilder.state_machine_def_from_dict(name, data, self.get_state_machine_names())

    def get_state_execute_order(self, state_machine_name: str,
                                sorted_state_defs: Optional[List[Tuple[str, BaseStateDefinition]]] = None) \
            -> List[Tuple[str, BaseStateDefinition]]:
        if sorted_state_defs is None:
            sorted_state_defs = []

        visited: Set[str] = set()
        stack: List[Tuple[str, BaseStateDefinition]] = []
        state_machine_def = self.load(state_machine_name)
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
                        and states_def.task_type.name == TaskType.STATE_MACHINE_PLAN.name):
                    return name, states_def

    def get_root_name(self) -> str:
        for definition in self.definitions.values():
            if definition.get("is_root"):
                return definition["name"]
