import json
import os
from typing import Dict, Any, Set

from thinker_ai.status_machine.state_machine import (ActionFactory, State, Transition, StateMachineDefinition,
                                                     StateMachineDefinitionRepository, CompositeState)


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
        return self._definition_from_dict(data)

    @staticmethod
    def _definition_to_dict(definition: StateMachineDefinition) -> Dict[str, Any]:
        return {
            "id": definition.id,
            "states": [FileBasedStateMachineDefinitionRepository._state_definition_to_dict(sd) for sd in definition.states],
            "transitions": [FileBasedStateMachineDefinitionRepository._transition_to_dict(t) for t in definition.transitions]
        }

    @staticmethod
    def _state_definition_to_dict(state: State) -> Dict[str, Any]:
        actions = [{"name": a.command, "command": a.command} for a in state.actions]
        return {
            "id": state.id,
            "name": state.name,
            "actions": actions,
            "events": list(state.events),
            "type": state.type,
            "inner_state_machine_definition_id": state.inner_state_machine_definition_id if isinstance(state, CompositeState) else None
        }

    @staticmethod
    def _transition_to_dict(transition: Transition) -> Dict[str, str]:
        return {
            "event": transition.event,
            "source": transition.source.id,
            "target": transition.target.id
        }

    def _definition_from_dict(self, data: Dict[str, Any]) -> StateMachineDefinition:
        states = {self._state_from_dict(sd) for sd in data["states"]}
        transitions = {self._transition_from_dict(t, states) for t in data["transitions"]}
        return StateMachineDefinition(id=data["id"], states=states, transitions=transitions)

    def _state_from_dict(self, data: Dict[str, Any]) -> State:
        actions = {ActionFactory.create_action(a) for a in data["actions"]}
        events = set(data["events"])
        if data.get("inner_state_machine_definition_id"):
            return CompositeState(
                id=data["id"],
                name=data["name"],
                actions=actions,
                events=events,
                type=data["type"],
                outer_to_inner_command=data.get("outer_to_inner_command"),
                inner_to_outer_event=data.get("inner_to_outer_event"),
                inner_state_machine_definition_id=data["inner_state_machine_definition_id"]
            )
        return State(id=data["id"], name=data["name"], actions=actions, result_events=events, type=data["type"])

    @staticmethod
    def _transition_from_dict(data: Dict[str, str], states: Set[State]) -> Transition:
        source = next(sd for sd in states if sd.id == data["source"])
        target = next(sd for sd in states if sd.id == data["target"])
        return Transition(event=data["event"], source=source, target=target)
