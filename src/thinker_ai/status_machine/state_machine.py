import importlib
import uuid
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, Set, cast, TypeVar, Type

T = TypeVar('T')


def from_class_name(cls: Type[T], class_name: str, **kwargs: Any) -> T:
    module_name, class_name = class_name.rsplit('.', 1)
    context_class = getattr(importlib.import_module(module_name), class_name)
    instance = context_class(**kwargs)
    if not isinstance(instance, cls):
        raise TypeError(f"Class {class_name} is not a subclass of {cls.__name__}")
    return instance


class Event:
    def __init__(self, name: str, id: Optional[str] = str(uuid.uuid4()), publisher_id: Optional[str] = None,
                 payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.publisher_id = publisher_id
        self.payload = payload


class Command:
    def __init__(self, name: str, target: str, id: Optional[str] = str(uuid.uuid4()), payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.target = target
        self.payload = payload


class Action(ABC):
    def __init__(self, on_command: str):
        self.on_command = on_command

    @abstractmethod
    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        raise NotImplementedError

    @classmethod
    def from_class_name(cls, class_name: str, **kwargs) -> 'Action':
        return from_class_name(cls, class_name, **kwargs)

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"


class CompositeAction(Action):
    def __init__(self, on_command: str):
        super().__init__(on_command)

    @abstractmethod
    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> Optional[Event]:
        # 子类实现该方法，分解为多个inner_command并根据需要调用该方法owner_state_context.handle_single_inner_command,最后通过to_outer_event返回结果
        raise NotImplementedError


class BaseStateDefinition:
    def __init__(self, id: str,
                 name: str,
                 description: str,
                 state_context_class_name: str
                 ):
        self.id = id
        self.name = name
        self.description = description
        self.state_context_class_name = state_context_class_name


class StateDefinition(BaseStateDefinition):
    def __init__(self, id: str, name: str, description: str, task_type: str, state_context_class_name: str,
                 actions: Set[Action], result_events: Set[str], is_start: bool = False):
        super().__init__(id, name, description, state_context_class_name)
        self.task_type = task_type
        self.is_start = is_start
        self.events: Set[str] = result_events
        self.actions: Set[Action] = actions


class BaseStateContext:
    def __init__(self, id: str, state_def: BaseStateDefinition):
        self.id = id
        self.state_def = state_def

    @classmethod
    def from_class_name(cls, class_name: str, **kwargs) -> 'BaseStateContext':
        return from_class_name(cls, class_name, **kwargs)

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any):
        raise NotImplementedError

    def get_state_def(self) -> BaseStateDefinition:
        return self.state_def

    def set_state_def(self, value: BaseStateDefinition):
        self.state_def = value


class StateContext(BaseStateContext):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)

    @classmethod
    def from_class_name(cls, class_name: str, **kwargs) -> 'StateContext':
        return from_class_name(cls, class_name, **kwargs)

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, on_command: str) -> Optional[Action]:
        for action in self.state_def.actions:
            if action.on_command == on_command:
                return action
        return None

    def assert_event(self, event: Event):
        if not event:
            raise RuntimeError(f'Event is None')
        if event.name not in self.state_def.events:
            raise RuntimeError(f'Illegal event "{event.name}" for state {self.state_def.name}')

    def handle(self, command: Command, outer_state_machine: 'StateMachine', **kwargs) -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command, self, **kwargs)
            if event:
                self.assert_event(event)
                outer_state_machine.on_event(event)
            return event
        else:
            raise Exception(
                f"Illegal command {command.name} for state {self.state_def.name}")


class Transition:
    def __init__(self, event: str, source: BaseStateDefinition, target: BaseStateDefinition):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self, id: str, states_def: Set[BaseStateDefinition], transitions: Set[Transition]):
        self.id = id
        self.states_def: Set[BaseStateDefinition] = states_def
        self.transitions: Set[Transition] = transitions

    def add_state_def(self, state_def: StateDefinition):
        self.states_def.add(state_def)

    def add_transition(self, transition: Transition):
        self.transitions.add(transition)

    def get_state_def(self, name: str) -> Optional[BaseStateDefinition]:
        for state in self.states_def:
            if state.name == name:
                return state
        return None

    def get_start_state_def(self) -> Optional[StateDefinition]:
        for state in self.states_def:
            if isinstance(state, StateDefinition) and state.is_start:
                return state
        return None


class InnerStateMachineDefinition(StateMachineDefinition):
    def __init__(self, id: str, states_def: Set[BaseStateDefinition], transitions: Set[Transition],
                 inner_state_to_outer_event: Optional[Dict[str, str]] = None
                 ):
        super().__init__(id, states_def, transitions)
        self.inner_state_to_outer_event = inner_state_to_outer_event if inner_state_to_outer_event is not None else {}

    def to_outer_event(self, inner_event: Event, state_context: BaseStateContext) -> Optional[Event]:
        if type(state_context.state_def) is not BaseStateDefinition:
            return None
        outer_event_name = self.inner_state_to_outer_event.get(state_context.state_def.name)
        if outer_event_name is None:
            return None
        return Event(id=inner_event.id, name=outer_event_name, payload=inner_event.payload)


class CompositeStateDefinition(StateDefinition):
    def __init__(self, id: str, name: str, description: str, task_type: str,
                 state_context_class_name: str,
                 actions: Set[Action],
                 events: Set[str],
                 inner_state_machine_definition: InnerStateMachineDefinition,
                 is_start
                 ):
        super().__init__(id, name, description, task_type, state_context_class_name, actions, events, is_start)
        self.inner_state_machine_definition = inner_state_machine_definition

    def to_outer_event(self, inner_event: Event, state_context: StateContext) -> Optional[Event]:
        return self.inner_state_machine_definition.to_outer_event(inner_event, state_context)


class StateMachineDefinitionRepository(ABC):
    @abstractmethod
    def load(self, id: str) -> StateMachineDefinition:
        raise NotImplementedError

    @abstractmethod
    def save(self, definition: StateMachineDefinition):
        raise NotImplementedError


class StateContextBuilder:
    def __init__(self, state_machine_context_repository: "StateMachineRepository",
                 state_machine_definition_repository: "StateMachineDefinitionRepository"):
        self.state_machine_context_repository = state_machine_context_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    def build(self, state_def: BaseStateDefinition,
              id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        if type(state_def) is CompositeStateDefinition:
            class_name = state_def.state_context_class_name
            if not class_name:
                class_name = "thinker_ai.status_machine.state_machine.CompositeStateDefinition"
            return CompositeStateContext.from_class_name(id=id,
                                                         class_name=class_name,
                                                         composite_state_def=state_def,
                                                         state_context_builder=self,
                                                         state_machine_repository=self.state_machine_context_repository,
                                                         state_machine_definition_repository=self.state_machine_definition_repository
                                                         )
        elif type(state_def) is StateDefinition:
            class_name = state_def.state_context_class_name
            if not class_name:
                class_name = "thinker_ai.status_machine.state_machine.StateContext"
            return StateContext.from_class_name(id=id,
                                                class_name=class_name,
                                                state_def=state_def)
        elif type(state_def) is BaseStateDefinition:
            class_name = state_def.state_context_class_name
            if not class_name:
                class_name = "thinker_ai.status_machine.state_machine.BaseStateContext"
            return BaseStateContext.from_class_name(id=id,
                                                    class_name=class_name,
                                                    state_def=state_def)


class StateMachine:
    def __init__(self, id: str,
                 state_machine_def_id: str,
                 current_state_context: BaseStateContext,
                 state_context_builder: StateContextBuilder,
                 state_machine_repository: "StateMachineRepository",
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 history: Optional[List[BaseStateContext]] = None):
        self.id = id
        self.state_context_builder = state_context_builder
        self.state_machine_def_id = state_machine_def_id
        self.current_state_context: BaseStateContext = current_state_context
        self.history: List[BaseStateContext] = history or []
        self.state_machine_repository: StateMachineRepository = state_machine_repository
        self.state_machine_definition_repository: StateMachineDefinitionRepository = state_machine_definition_repository

    def handle(self, command: Command, **kwargs) -> Optional[Event]:
        if self.id != command.target:
            raise Exception(f"the state context {self.id} do not the command  target {command.target}")
        if isinstance(self.current_state_context, StateContext):
            return self.current_state_context.handle(command, self, **kwargs)
        else:
            raise Exception(
                f"Illegal command {command.name} for state {self.current_state_context.state_def.name}")

    def on_event(self, event: Event):
        if not self.state_machine_definition_repository:
            raise ValueError("Repositories are not set")
        state_machine_definition = self.get_state_machine_def()
        transitions = state_machine_definition.transitions
        for transition in transitions:
            if transition.event == event.name and transition.source.name == self.current_state_context.state_def.name:
                self.history.append(self.current_state_context)
                self.current_state_context = self.get_state_context(transition.target)
                self.state_machine_repository.save(self)
                return
        raise ValueError(
            f"No transition from state '{self.current_state_context.state_def.name}' with event '{event.name}'")

    def get_state_machine_def(self) -> StateMachineDefinition:
        return self.state_machine_definition_repository.load(self.state_machine_def_id)

    def last_state(self) -> Optional[StateContext]:
        return self.history[-1] if self.history else None

    def get_state_context(self, target: BaseStateDefinition) -> BaseStateContext:
        state_context = next((sc for sc in self.history if sc.state_def.name == target.name), None)
        if not state_context:
            state_context = self.state_context_builder.build(target)
        return state_context

    def to_outer_event(self, inner_event) -> Optional[Event]:
        state_machine_def = self.get_state_machine_def()
        if isinstance(state_machine_def, InnerStateMachineDefinition):
            return state_machine_def.to_outer_event(inner_event, self.current_state_context)


class StateMachineRepository(ABC):
    @abstractmethod
    def load(self, id: str) -> StateMachine:
        raise NotImplementedError

    @abstractmethod
    def save(self, state_machine_context: StateMachine):
        raise NotImplementedError


class CompositeStateContext(StateContext):
    def __init__(self, id: str,
                 composite_state_def: CompositeStateDefinition,
                 state_context_builder: StateContextBuilder,
                 state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, composite_state_def)
        self.state_context_builder = state_context_builder
        self.state_machine_repository = state_machine_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    @classmethod
    def from_class_name(cls, class_name: str, **kwargs) -> 'CompositeStateContext':
        return from_class_name(cls, class_name, **kwargs)

    def handle(self, command: Command, outer_state_machine: 'StateMachine', **kwargs) -> Optional[Event]:
        action = self.get_action(command.name)
        if action:
            event = action.handle(command, self, **kwargs)
            if event:
                self.assert_event(event)
                outer_state_machine.on_event(event)
            return event

    def handle_inner(self, inner_command: Command, **kwargs) -> Optional[Event]:
        inner_event = self.get_state_machine().handle(inner_command, **kwargs)
        return self.to_outer_event(inner_event)

    def get_state_def(self) -> CompositeStateDefinition:
        return cast(CompositeStateDefinition, self.state_def)

    def set_state_def(self, value: CompositeStateDefinition):
        self.state_def = value

    def get_state_machine(self) -> StateMachine:
        my_state_machine = self.state_machine_repository.load(self.id)
        if not my_state_machine:
            state_machine_definition = self.state_machine_definition_repository.load(
                self.get_state_def().id)
            my_state_machine = StateMachine(id=self.id,
                                            state_machine_def_id=self.get_state_def().id,
                                            current_state_context=self.state_context_builder.build(
                                                state_machine_definition.get_start_state_def()),
                                            state_context_builder=self.state_context_builder,
                                            state_machine_repository=self.state_machine_repository,
                                            state_machine_definition_repository=self.state_machine_definition_repository,
                                            history=[]
                                            )
            self.state_machine_repository.save(my_state_machine)
        return my_state_machine

    def get_action(self, on_command: str) -> Optional[CompositeAction]:
        for action in self.state_def.actions:
            if action.on_command == on_command and isinstance(action, CompositeAction):
                return action
        return None

    def to_outer_event(self, inner_event: Event) -> Optional[Event]:
        if inner_event:
            return self.get_state_machine().to_outer_event(inner_event)


class ActionFactory:
    _registry = {}

    @classmethod
    def register_action(cls, action_class_name: str, action_cls: type):
        cls._registry[action_class_name] = action_cls

    @classmethod
    def create_action(cls, action_class_name) -> Optional[Action]:
        action_cls: Action = cls._registry.get(action_class_name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for class '{action_class_name}'")
        return action_cls.from_class_name(action_class_name)