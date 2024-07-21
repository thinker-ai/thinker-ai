import uuid
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, Set, Literal, cast


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
    def __init__(self, command: str):
        self.command = command

    @abstractmethod
    def handle(self, command: Command, state_context: "StateContext", **kwargs) -> Optional[Event]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> 'Action':
        raise NotImplementedError


StateType = Literal["start", "middle", "end"]


class StateDefinition:
    def __init__(self, id: str, name: str, actions: Set[Action], result_events: Set[str], type: StateType):
        self.id = id
        self.name = name
        self.type = type
        self.events: Set[str] = result_events
        self.actions: Set[Action] = actions


class StateContext:
    def __init__(self, id: str, state_def: StateDefinition):
        self.id = id
        self.state_def = state_def

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any):
        raise NotImplementedError

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, command: str) -> Optional[Action]:
        for action in self.state_def.actions:
            if action.command == command:
                return action
        return None

    def assert_event(self, event: Event):
        if not event:
            raise RuntimeError(f'Event is None')
        if event.name not in self.state_def.events:
            raise RuntimeError(f'Illegal event "{event.name}" for state {self.state_def.name}')

    def handle(self, command: Command, outer_state_machine: 'StateMachineContext', **kwargs) -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command, self, **kwargs)
            if event:
                self.assert_event(event)
                outer_state_machine.on_event(event)
            return event
        return None


class Transition:
    def __init__(self, event: str, source: StateDefinition, target: StateDefinition):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self, id: str, states_def: Set[StateDefinition], transitions: Set[Transition]):
        self.id = id
        self.states_def: Set[StateDefinition] = states_def
        self.transitions: Set[Transition] = transitions

    def add_state_def(self, state_def: StateDefinition):
        self.states_def.add(state_def)

    def add_transition(self, transition: Transition):
        self.transitions.add(transition)

    def get_state_def(self, name: str) -> Optional[StateDefinition]:
        for state in self.states_def:
            if state.name == name:
                return state
        return None

    def get_start_state_def(self) -> Optional[StateDefinition]:
        for state in self.states_def:
            if state.type == 'start':
                return state
        return None


class InnerStateMachineDefinition(StateMachineDefinition):
    def __init__(self, id: str, states_def: Set[StateDefinition], transitions: Set[Transition],
                 outer_to_inner_command: Optional[Dict[str, str]] = None,
                 inner_to_outer_event: Optional[Dict[str, str]] = None
                 ):
        super().__init__(id, states_def, transitions)
        self.outer_to_inner_command = outer_to_inner_command if outer_to_inner_command is not None else {}
        self.inner_to_outer_event = inner_to_outer_event if inner_to_outer_event is not None else {}

    def to_inner_command(self, command: Command, owner_state_context_id: str) -> Optional[Command]:
        inner_command_name = self.outer_to_inner_command.get(command.name)
        if inner_command_name is None:
            return None
        return Command(id=command.id, name=inner_command_name, payload=command.payload,
                       target=owner_state_context_id)

    def from_inner_event(self, inner_event: Event) -> Optional[Event]:
        outer_event_name = self.inner_to_outer_event.get(inner_event.name)
        if outer_event_name is None:
            return None
        return Event(id=inner_event.id, name=outer_event_name, payload=inner_event.payload)


class StateMachineDefinitionRepository(ABC):
    @abstractmethod
    def load(self, id: str) -> StateMachineDefinition:
        raise NotImplementedError

    @abstractmethod
    def save(self, definition: StateMachineDefinition):
        raise NotImplementedError


class StateContextBuilder:
    def __init__(self, state_machine_context_repository: "StateMachineContextRepository",
                 state_machine_definition_repository: "StateMachineDefinitionRepository"):
        self.state_machine_context_repository = state_machine_context_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    def build(self, state_def: StateDefinition, id: Optional[str] = None) -> StateContext:
        if id is None:
            id = str(uuid.uuid4())
        if not isinstance(state_def, CompositeStateDefinition):
            return StateContext(id, state_def)
        else:
            return CompositeStateContext(id=id,
                                         composite_state_def=state_def,
                                         state_context_builder=self,
                                         state_machine_context_repository=self.state_machine_context_repository,
                                         state_machine_definition_repository=self.state_machine_definition_repository
                                         )


class StateMachineContext:
    def __init__(self, id: str,
                 definition_id: str,
                 current_context: StateContext,
                 state_context_builder: StateContextBuilder,
                 state_machine_context_repository: "StateMachineContextRepository",
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 history: Optional[List[StateContext]] = None):
        self.id = id
        self.state_context_builder = state_context_builder
        self.definition_id = definition_id
        self.current_context: StateContext = current_context
        self.history: List[StateContext] = history or []
        self.state_machine_context_repository: StateMachineContextRepository = state_machine_context_repository
        self.state_machine_definition_repository: StateMachineDefinitionRepository = state_machine_definition_repository

    def handle(self, command: Command, **kwargs) -> Optional[Event]:
        if self.id == command.target or isinstance(self.current_context, CompositeStateContext):
            return self.current_context.handle(command, self, **kwargs)

    def on_event(self, event: Event):
        if not self.state_machine_definition_repository:
            raise ValueError("Repositories are not set")
        state_machine_definition = self.state_machine_definition_repository.load(
            self.definition_id)
        transitions = state_machine_definition.transitions
        for transition in transitions:
            if transition.event == event.name and transition.source.name == self.current_context.state_def.name:
                self.history.append(self.current_context)
                self.current_context = self.get_state_context(transition.target)
                self.state_machine_context_repository.save(self)
                return
        raise ValueError(
            f"No transition from state '{self.current_context.state_def.name}' with event '{event.name}'")

    def last_state(self) -> Optional[StateContext]:
        return self.history[-1] if self.history else None

    def get_state_context(self, target: StateDefinition) -> StateContext:
        state_context = next((sc for sc in self.history if sc.state_def.name == target.name), None)
        if not state_context:
            state_context = self.state_context_builder.build(target)
        return state_context


class StateMachineContextRepository(ABC):
    @abstractmethod
    def load(self, id: str) -> StateMachineContext:
        raise NotImplementedError

    @abstractmethod
    def save(self, state_machine_context: StateMachineContext):
        raise NotImplementedError


class CompositeStateDefinition(StateDefinition):
    def __init__(self, id: str, name: str,
                 actions: Set[Action],
                 events: Set[str],
                 type: StateType,
                 inner_state_machine_definition: InnerStateMachineDefinition
                 ):
        super().__init__(id, name, actions, events, type)
        self.inner_state_machine_definition = inner_state_machine_definition

    def to_inner_command(self, command: Command, owner_state_context_id: str) -> Optional[Command]:
        return self.inner_state_machine_definition.to_inner_command(command, owner_state_context_id)

    def from_inner_event(self, inner_event: Event) -> Optional[Event]:
        return self.inner_state_machine_definition.from_inner_event(inner_event)


class CompositeStateContext(StateContext):
    def __init__(self, id: str,
                 composite_state_def: CompositeStateDefinition,
                 state_context_builder: StateContextBuilder,
                 state_machine_context_repository: StateMachineContextRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, composite_state_def)
        self.state_context_builder = state_context_builder
        self.state_machine_context_repository = state_machine_context_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    def handle(self, command: Command, outer_state_machine: 'StateMachineContext', **kwargs) -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command, self, **kwargs)
        else:
            event = self._handle_inner(command, **kwargs)
        if event:
            self.assert_event(event)
            outer_state_machine.on_event(event)
        return event

    def get_state_def(self) -> CompositeStateDefinition:
        return cast(CompositeStateDefinition, self.state_def)

    def set_state_def(self, value: CompositeStateDefinition):
        self.state_def = value

    def get_inner_state_machine_context(self) -> StateMachineContext:
        inner_state_machine_context = self.state_machine_context_repository.load(self.id)
        if not inner_state_machine_context:
            state_machine_definition = self.state_machine_definition_repository.load(
                self.get_state_def().id)
            inner_state_machine_context = StateMachineContext(id=self.id,
                                                              definition_id=self.get_state_def().id,
                                                              current_context=self.state_context_builder.build(
                                                                  state_machine_definition.get_start_state_def()),
                                                              state_context_builder=self.state_context_builder,
                                                              state_machine_context_repository=self.state_machine_context_repository,
                                                              state_machine_definition_repository=self.state_machine_definition_repository,
                                                              history=[]
                                                              )
            self.state_machine_context_repository.save(inner_state_machine_context)
        return inner_state_machine_context

    def _handle_inner(self, command: Command, **kwargs):
        inner_state_machine_context = self.get_inner_state_machine_context()
        inner_current_state_context = inner_state_machine_context.current_context
        inner_command = self.get_state_def().to_inner_command(command, self.id)
        if inner_command:
            # 外部间接调用内部命令
            inner_event = inner_current_state_context.handle(inner_command, inner_state_machine_context, **kwargs)
        else:
            # 外部直接调用内部命令
            inner_event = inner_current_state_context.handle(command, inner_state_machine_context, **kwargs)
        if inner_event:
            return self.get_state_def().from_inner_event(inner_event)
        return None


class ActionFactory:
    _registry = {}

    @classmethod
    def register_action(cls, action_type: str, action_cls: type):
        cls._registry[action_type] = action_cls

    @classmethod
    def create_action(cls, action_data) -> Optional[Action]:
        name = action_data.get('name')
        action_cls = cls._registry.get(name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for type '{name}'")
        return action_cls.from_dict({"command": action_data.get('command', '')})
