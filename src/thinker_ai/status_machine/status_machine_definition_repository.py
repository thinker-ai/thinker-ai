import json
import os
from typing import Dict, Any, Set

from thinker_ai.status_machine.state_machine import (ActionFactory, StateDefinition, Transition, StateMachineDefinition,
                                                     StateMachineDefinitionRepository, BaseStateDefinition)


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

    def save_json_text(self, json_text):
        definitions: dict = json.loads(json_text)
        id, definition = list(definitions.items())[0]
        self.save_json(id, definition)

    def save(self, definition: StateMachineDefinition):
        self.definitions[definition.id] = self._definition_to_dict(definition)
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def save_json(self, id: str, definition: dict):
        self.definitions[id] = definition
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.definitions, file, indent=2, ensure_ascii=False)

    def load(self, id: str) -> StateMachineDefinition:
        if id not in self.definitions:
            raise ValueError(f"StateMachineDefinition with id '{id}' not found")

        data = self.definitions[id]
        return self._definition_from_dict(id, data)

    def _definition_from_dict(self, id: str, data: Dict[str, Any]) -> StateMachineDefinition:
        states = {self._state_from_dict(sd) for sd in data["states_def"]}
        transitions = {self._transition_from_dict(t, states) for t in data["transitions"]}
        return StateMachineDefinition(
            id=id,
            name=data["name"],
            states_def=states,
            transitions=transitions,
            inner_end_state_to_outer_event=data.get("inner_end_state_to_outer_event")
        )

    @staticmethod
    def _definition_to_dict(definition: StateMachineDefinition) -> Dict[str, Any]:
        result = {
            "id": definition.id,
            "name":definition.name,
            "states_def": [FileBasedStateMachineDefinitionRepository._state_definition_to_dict(sd) for sd in
                           definition.states_def],
            "transitions": [FileBasedStateMachineDefinitionRepository._transition_to_dict(t) for t in
                            definition.transitions],
        }
        if definition.inner_end_state_to_outer_event:
            result["inner_end_state_to_outer_event"] = definition.inner_end_state_to_outer_event
        return result

    @staticmethod
    def _state_definition_to_dict(state_def: BaseStateDefinition) -> Dict[str, Any]:
        if isinstance(state_def, StateDefinition):
            actions = [{a.on_command: a.get_full_class_name()} for a in state_def.actions]
            return {
                "id": state_def.id,
                "name": state_def.name,
                "description": state_def.description,
                "state_context_class_name": state_def.state_context_class_name,
                "task_type": state_def.task_type,
                "actions": actions,
                "events": list(state_def.events),
                "is_start": state_def.is_start
            }
            # state_def包含的inner_state_machine_definition信息处于它的下级state_dict中，所以不设置到本级的state_dict中
            # if isinstance(state_def, CompositeStateDefinition):
            #     state_dict["inner_state_machine_definition"] = state_def.inner_state_machine_definition.id
        elif isinstance(state_def, BaseStateDefinition):
            return {
                "id": state_def.id,
                "name": state_def.name,
                "description": state_def.description,
                "state_context_class_name": state_def.state_context_class_name
            }

    @staticmethod
    def _transition_to_dict(transition: Transition) -> Dict[str, str]:
        return {
            "event": transition.event,
            "source": transition.source.id,
            "target": transition.target.id
        }

    def _state_from_dict(self, data: Dict[str, Any]) -> BaseStateDefinition:
        if data.get("actions"):
            return StateDefinition(
                id=data["id"],
                name=data["name"],
                state_context_class_name=data["state_context_class_name"],
                description=data.get("description", ""),
                task_type=data.get("task_type", ""),
                actions={ActionFactory.create_action(a) for a in data["actions"]},
                result_events=set(data["events"]),
                inner_state_machine_definition=self.load(data["id"]) if data["id"] in self.definitions else None,
                is_start=data.get("is_start")
            )
        else:
            return BaseStateDefinition(id=data["id"],
                                       name=data["name"],
                                       description=data.get("description", ""),
                                       state_context_class_name=data.get("state_context_class_name")
                                       )

    @staticmethod
    def _transition_from_dict(data: Dict[str, str], states: Set[BaseStateDefinition]) -> Transition:
        source = next(sd for sd in states if sd.id == data["source"])
        target = next(sd for sd in states if sd.id == data["target"])
        return Transition(event=data["event"], source=source, target=target)
