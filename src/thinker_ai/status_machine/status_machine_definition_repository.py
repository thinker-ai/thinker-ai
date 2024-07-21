import json
import os
from typing import Dict, Any, Set, cast

from thinker_ai.status_machine.state_machine import (ActionFactory, StateDefinition, Transition, StateMachineDefinition,
                           StateMachineDefinitionRepository, InnerStateMachineDefinition, CompositeStateDefinition)


class FileBasedStateMachineDefinitionRepository(StateMachineDefinitionRepository):
    def __init__(self, base_dir: str, file_name: str):
        self.base_dir = base_dir
        self.file_path = os.path.join(base_dir, file_name)
        self.definitions = self._load_definitions()

    def _load_definitions(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        return {}

    def save(self, definition: StateMachineDefinition):
        self.definitions[definition.id] = self._definition_to_dict(definition)
        with open(self.file_path, 'w') as file:
            json.dump(self.definitions, file, indent=2)

    def load(self, id: str) -> StateMachineDefinition:
        if id not in self.definitions:
            raise ValueError(f"StateMachineDefinition with id '{id}' not found")

        data = self.definitions[id]
        return self._definition_from_dict(id, data)

    def _definition_from_dict(self, id: str, data: Dict[str, Any]) -> StateMachineDefinition:
        states = {self._state_from_dict(sd) for sd in data["states_def"]}
        transitions = {self._transition_from_dict(t, states) for t in data["transitions"]}
        if data.get("outer_state_machine_mapping"):
            return InnerStateMachineDefinition(
                id=id,
                states_def=states,
                transitions=transitions,
                outer_to_inner_command=data["outer_state_machine_mapping"]["outer_to_inner_command"],
                inner_to_outer_event=data["outer_state_machine_mapping"]["inner_to_outer_event"]
            )
        return StateMachineDefinition(id=id, states_def=states, transitions=transitions)

    @staticmethod
    def _definition_to_dict(definition: StateMachineDefinition) -> Dict[str, Any]:
        result = {
            "states_def": [FileBasedStateMachineDefinitionRepository._state_definition_to_dict(sd) for sd in
                           definition.states_def],
            "transitions": [FileBasedStateMachineDefinitionRepository._transition_to_dict(t) for t in
                            definition.transitions]
        }
        if isinstance(definition, InnerStateMachineDefinition):
            result["outer_state_machine_mapping"] = {
                "outer_to_inner_command": definition.outer_to_inner_command,
                "inner_to_outer_event": definition.inner_to_outer_event
            }
        return result

    @staticmethod
    def _state_definition_to_dict(state_def: StateDefinition) -> Dict[str, Any]:
        actions = [{"name": a.command, "command": a.command} for a in state_def.actions]
        return {
            "id": state_def.id,
            "name": state_def.name,
            "actions": actions,
            "events": list(state_def.events),
            "type": state_def.type
        }

    @staticmethod
    def _transition_to_dict(transition: Transition) -> Dict[str, str]:
        return {
            "event": transition.event,
            "source": transition.source.id,
            "target": transition.target.id
        }

    def _state_from_dict(self, data: Dict[str, Any]) -> StateDefinition:
        actions = {ActionFactory.create_action(a) for a in data["actions"]}
        events = set(data["events"])
        # 检查是否存在对应的子状态机定义
        if data["id"] in self.definitions:
            inner_state_machine = cast(InnerStateMachineDefinition, self.load(data["id"]))
            return CompositeStateDefinition(
                id=data["id"],
                name=data["name"],
                actions=actions,
                events=events,
                type=data["type"],
                inner_state_machine_definition=inner_state_machine
            )
        return StateDefinition(id=data["id"], name=data["name"], actions=actions, result_events=events,
                               type=data["type"])

    @staticmethod
    def _transition_from_dict(data: Dict[str, str], states: Set[StateDefinition]) -> Transition:
        source = next(sd for sd in states if sd.id == data["source"])
        target = next(sd for sd in states if sd.id == data["target"])
        return Transition(event=data["event"], source=source, target=target)
