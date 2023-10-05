from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any

from pydantic import BaseModel


class Event(BaseModel):
    name: str
    event_id: Optional[str]
    publisher_id: Optional[str]
    payload: Optional[Any]


class Command(BaseModel):
    name: str
    command_id: Optional[str]
    command_id: Optional[str]
    payload: Optional[Any]


class Action(ABC):
    @abstractmethod
    def handle(self, command: Command) -> Optional[Event]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> 'Action':
        raise NotImplementedError


class State:
    def __init__(self, name: str, actions: Dict[str, Action]):
        self.name = name
        self.actions: Dict[str, Action] = actions

    def handle(self, command: Command, state_machine: 'StateMachine') -> Optional[Event]:
        action = self.actions.get(command.name)
        if action:
            event = action.handle(command)
            if event:
                state_machine.on_event(event)
                return event
        return None


class Transition:
    def __init__(self, event: str, source: State, target: State):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self, business_id: str, states: Dict[str, State], transitions: Dict[str, List[Transition]]):
        self.business_id = business_id
        self.states: Dict[str, State] = states
        self.transitions: Dict[str, List[Transition]] = transitions

    def add_state(self, state: State):
        self.states[state.name] = state

    def add_transition(self, transition: Transition):
        self.transitions.setdefault(transition.event, []).append(transition)

    def get_state(self, name: str) -> State:
        return self.states.get(name)


class StateMachine:
    def __init__(self, definition: StateMachineDefinition, instance_id, current_state, history):
        self.definition = definition
        self.history: List[State] = history
        self.instance_id = instance_id
        self.current_state = current_state

    def handle(self, command: Command):
        return self.current_state.handle(command, self)

    def on_event(self, event: Event):
        transitions = self.definition.transitions.get(event.name, [])
        for transition in transitions:
            if transition.source == self.current_state:
                self.history.append(self.current_state)
                self.current_state = transition.target
                return
        raise ValueError(f"No transition from state '{self.current_state.name}' with event '{event.name}'")

    def last_state(self) -> Optional[State]:
        return self.history[-1] if self.history else None


class ActionFactory:
    _registry = {}

    @classmethod
    def register_action(cls, action_type, action_cls):
        cls._registry[action_type] = action_cls

    @classmethod
    def create_action(cls, action_name, data):
        action_cls = cls._registry.get(action_name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for type '{action_name}'")
        return action_cls.from_dict(data)
