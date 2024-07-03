import json
from abc import ABC, abstractmethod
from typing import List, Dict

from thinker_ai.status_machine.state_machine import StateMachine, Transition, StateMachineDefinition, State, \
    ActionFactory


class StateMachineDefinitionAbstractPersistence(ABC):
    @abstractmethod
    def save(self, definition_id: str, data: str) -> None:
        pass

    @abstractmethod
    def load(self, definition_id: str) -> str:
        pass


class StateMachineAbstractPersistence(ABC):

    @abstractmethod
    def save(self, definition_id: str, instance_id: str, data: str) -> None:
        pass

    @abstractmethod
    def load(self, definition_id: str, instance_id: str) -> str:
        pass


class StateMachineDefinitionRepository:
    def __init__(self, persistence: StateMachineDefinitionAbstractPersistence):
        self.persistence = persistence

    @staticmethod
    def _deserialize(json_str: str) -> StateMachineDefinition:
        data = json.loads(json_str)
        id = data.get("id", "")
        states: Dict = {}
        for state_name, actions_data in data.get("states", {}).items():
            actions = {}
            for action_type, action_data in actions_data.items():
                action_type = action_data['type']
                actions[action_type] = ActionFactory.create_action(action_type, action_data)
            state = State(name=state_name, actions=actions,state_machine_definition_id=id)
            states[state_name] = state
        transitions_list: List[Dict[str, str]] = data.get("transitions")
        transitions: Dict = {}
        for transition_data in transitions_list:
            event = transition_data.get("event")
            if event:
                transition = Transition(
                    event,
                    states.get(transition_data.get("from")),
                    states.get(transition_data.get("to"))
                )
                transitions.setdefault(event, []).append(transition)
        return StateMachineDefinition(id, states, transitions)

    @staticmethod
    def _serialize(definition: StateMachineDefinition) -> str:
        data = {
            'id': definition.id,
            'states': {
                state_name: {
                    action_type: {
                        'type': action_type,
                    }
                    for action_type, action in state.actions.items()
                }
                for state_name, state in definition.states.items()
            },
            'transitions': [{
                'from': transition.source.command,
                'to': transition.target.command,
                'event': transition.event
            } for transitions in definition.transitions.values() for transition in transitions]
        }
        return json.dumps(data, ensure_ascii=False)

    def save(self, definition: StateMachineDefinition) -> None:
        name = definition.id
        data_str = self._serialize(definition)
        self.persistence.save(name, data_str)

    def load(self, name: str) -> StateMachineDefinition:
        data_str = self.persistence.load(name)
        if not data_str:
            raise ValueError(f"No data found for definition: {name}")
        return self._deserialize(data_str)


class StateMachineRepository:
    def __init__(self, persistence: StateMachineAbstractPersistence,
                 definition_repository: StateMachineDefinitionRepository):
        self.definition_repository = definition_repository
        self.persistence = persistence

    @staticmethod
    def _serialize(state_machine: StateMachine) -> str:
        data = {
            'instance_id': state_machine.id,
            'current_state': state_machine.current_state.name,
            'history': [state.name for state in state_machine.history]
        }
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize(state_machine_definition: StateMachineDefinition, data_str: str) -> StateMachine:
        data = json.loads(data_str)
        instance_id = data.get('instance_id', "")
        if not instance_id:
            raise ValueError("instance_id is None")
        current_state = state_machine_definition.get_state(data.get('current_state'))
        if not current_state:
            raise ValueError("current_state is None")
        history = [state_machine_definition.get_state(name) for name in data.get('history', [])]
        return StateMachine(state_machine_definition, instance_id, current_state, history)

    def save(self, state_machine: StateMachine):
        data_str = self._serialize(state_machine)
        self.persistence.save(state_machine.definition.id, state_machine.id, data_str)

    def load(self, id: str, instance_id: str) -> StateMachine:
        definition = self.definition_repository.load(id)
        data_str = self.persistence.load(id, instance_id)
        return self._deserialize(definition, data_str)
