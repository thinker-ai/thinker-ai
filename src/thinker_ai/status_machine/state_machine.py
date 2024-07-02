from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, cast, Set


class Event:
    def __init__(self, name: str, id: Optional[str] = None, publisher_id: Optional[str] = None,
                 payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.publisher_id = publisher_id
        self.payload = payload


class Command:
    def __init__(self, name: str, target: str, id: Optional[str] = None, payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.target = target
        self.payload = payload


class Action(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def handle(self, command: Command) -> Optional[Event]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> 'Action':
        raise NotImplementedError


class State:
    def __init__(self, name: str, actions: Set[Action]):
        self.name = name
        self.actions: Set[Action] = actions

    def get_action(self, name: str) -> Optional[Action]:
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def handle(self, command: Command, outer_state_machine: 'StateMachine') -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command)
            if event:
                outer_state_machine.on_event(event)
                return event
        return None


class Transition:
    def __init__(self, event: str, source: State, target: State):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self, id: str, states: Set[State], transitions: Set[Transition]):
        self.id = id
        self.states: Set[State] = states
        self.transitions: Set[Transition] = transitions

    def add_state(self, state: State):
        self.states.add(state)

    def add_transition(self, transition: Transition):
        self.transitions.add(transition)

    def get_state(self, name: str) -> Optional[State]:
        for stats in self.states:
            if stats.name == name:
                return stats
        return None


class StateMachine:
    def __init__(self, definition: StateMachineDefinition, id: str, current_state: State,
                 history: Optional[List[State]] = None):
        self.definition = definition
        self.history: List[State] = history or []
        self.id = id
        self.current_state = current_state

    def handle(self, command: Command) -> Optional[Event]:
        if (self.definition.id == command.target  # 现在能找到
                or isinstance(self.current_state, CompositeState)):  # 或进入内部寻找
            return self.current_state.handle(command, self)

    def on_event(self, event: Event):
        transitions = self.definition.transitions
        for transition in transitions:
            if transition.event == event.name and transition.source == self.current_state:
                self.history.append(self.current_state)
                self.current_state = transition.target
                return
        raise ValueError(f"No transition from state '{self.current_state.name}' with event '{event.name}'")

    def last_state(self) -> Optional[State]:
        return self.history[-1] if self.history else None


class CompositeState(State):
    def __init__(self, name: str, actions: Set[Action],
                 inner_state_machine: 'StateMachine',
                 to_inner_command_name_map: dict,
                 from_inner_event_name_map: dict
                 ):
        super().__init__(name, actions)
        self.inner_state_machine = inner_state_machine
        self.to_inner_command_name_map = to_inner_command_name_map
        self.from_inner_event_name_map = from_inner_event_name_map

    def handle(self, command: Command, outer_state_machine: 'StateMachine') -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command)
        else:
            event = self._handle_inner(command)
        if event:
            outer_state_machine.on_event(event)
            return event
        return None

    def _handle_inner(self, command: Command):
        inner_current_state = self.inner_state_machine.current_state
        inner_command = self.to_inner_command(command)
        if inner_command:
            inner_event = inner_current_state.handle(inner_command, self.inner_state_machine)
        else:
            inner_event = inner_current_state.handle(command, self.inner_state_machine)
        if inner_event:
            return self.from_inner_event(inner_event)
        return None

    def to_inner_command(self, command: Command) -> Optional[Command]:
        inner_command_name = self.to_inner_command_name_map.get(command.name)
        if inner_command_name is None:
            return None
        return Command(id=command.id, name=inner_command_name, payload=command.payload,
                       target=self.inner_state_machine.definition.id)

    def from_inner_event(self, inner_event: Event) -> Optional[Event]:
        outer_event_name = self.from_inner_event_name_map.get(inner_event.name)
        if outer_event_name is None:
            return None
        return Event(id=inner_event.id, name=outer_event_name, payload=inner_event.payload)


class ActionFactory:
    _registry = {}

    @classmethod
    def register_action(cls, action_type: str, action_cls: type):
        cls._registry[action_type] = action_cls

    @classmethod
    def create_action(cls, action_name: str, data: Dict) -> Action:
        action_cls = cls._registry.get(action_name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for type '{action_name}'")
        return action_cls.from_dict(data)
