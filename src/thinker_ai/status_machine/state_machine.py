import importlib
import json
import uuid
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, Set, cast, TypeVar, Type

from thinker_ai.status_machine.task_desc import TaskTypeDef, TaskType

T = TypeVar('T')


def from_class_name(cls: Type[T], full_class_name: str, **kwargs: Any) -> T:
    module_name, full_class_name = full_class_name.rsplit('.', 1)
    context_class = getattr(importlib.import_module(module_name), full_class_name)
    instance = context_class(**kwargs)
    if not isinstance(instance, cls):
        raise TypeError(f"Class {full_class_name} is not a subclass of {cls.__name__}")
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
    def from_class_name(cls, full_class_name: str, **kwargs) -> 'Action':
        return from_class_name(cls, full_class_name, **kwargs)

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


class ActionDescription:
    def __init__(self, data: dict):
        self.on_command = data["on_command"]
        self.full_class_name = data["full_class_name"]


class BaseStateDefinition:
    def __init__(self,
                 name: str,
                 description: str,
                 state_context_class_name: str
                 ):
        self.name = name
        self.description = description
        self.state_context_class_name = state_context_class_name


class StateDefinition(BaseStateDefinition):
    def __init__(self, name: str, description: str, task_type: TaskTypeDef, state_context_class_name: str,
                 actions_des: Set[ActionDescription], result_events: Set[str], is_start: bool = False,
                 is_composite: bool = False):
        super().__init__(name, description, state_context_class_name)
        self.task_type = task_type
        self.is_start = is_start
        self.events: Set[str] = result_events
        self.actions_des: Set[ActionDescription] = actions_des
        self.is_composite = is_composite


class BaseStateContext:
    def __init__(self, id: str, state_def: BaseStateDefinition):
        self.id = id
        self.state_def = state_def

    @classmethod
    def from_class_name(cls, full_class_name: str, **kwargs) -> 'BaseStateContext':
        return from_class_name(cls, full_class_name, **kwargs)

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def get_state_def(self) -> BaseStateDefinition:
        return self.state_def

    def set_state_def(self, value: BaseStateDefinition):
        self.state_def = value


class StateContext(BaseStateContext):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)

    @classmethod
    def from_class_name(cls, full_class_name: str, **kwargs) -> 'StateContext':
        return from_class_name(cls, full_class_name, **kwargs)

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, on_command: str) -> Optional[Action]:
        action_des = self.get_action_des(on_command)
        if action_des:
            return ActionFactory.create_action(action_des)
        return None

    def get_action_des(self, on_command: str) -> Optional[ActionDescription]:
        for action_des in self.state_def.actions_des:
            if action_des.on_command == on_command:
                return action_des
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
    def __init__(self,
                 name: str,
                 states_def: Set[BaseStateDefinition],
                 transitions: Set[Transition],
                 is_root: bool = False,
                 inner_end_state_to_outer_event: Optional[Dict[str, str]] = None):
        self.name = name
        self.is_root = is_root
        self.states_def: Set[BaseStateDefinition] = states_def
        self.transitions: Set[Transition] = transitions
        self.inner_end_state_to_outer_event = inner_end_state_to_outer_event if inner_end_state_to_outer_event is not None else {}

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

    def to_outer_event(self, inner_event: Event, state_context: BaseStateContext) -> Optional[Event]:
        if type(state_context.state_def) is not BaseStateDefinition:
            return None
        outer_event_name = self.inner_end_state_to_outer_event.get(state_context.state_def.name)
        if outer_event_name is None:
            return None
        return Event(id=inner_event.id, name=outer_event_name, payload=inner_event.payload)


class StateMachineDefinitionRepository(ABC):
    @abstractmethod
    def get(self, name: str) -> StateMachineDefinition:
        raise NotImplementedError

    @abstractmethod
    def set(self, name: str, definition: StateMachineDefinition):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError


class StateMachineDefinitionBuilder:

    def root_from_dict(self, state_machine_defs: dict) -> StateMachineDefinition:
        for name, state_machine_def in iter(state_machine_defs.items()):
            if state_machine_def.get("is_root"):
                return self.state_machine_def_from_dict(name, state_machine_def, set(state_machine_defs.keys()))

    def root_from_json(self, json_data: str) -> StateMachineDefinition:
        data = json.loads(json_data)
        return self.root_from_dict(data)

    def root_to_dict(self, root_state_machine_def: StateMachineDefinition) -> dict:
        if root_state_machine_def.is_root:
            return {root_state_machine_def.name: self.state_machine_def_to_dict(root_state_machine_def)}

    def root_to_json(self, root_state_machine_def: StateMachineDefinition) -> str:
        data = self.root_to_dict(root_state_machine_def)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def state_machine_def_from_dict(self, name: str, data: Dict[str, Any],
                                    exist_state_machine_names: Set) -> StateMachineDefinition:
        states = {self._state_def_from_dict(data=sd,
                                            state_machine_name=name,
                                            exist_state_machine_names=exist_state_machine_names)
                  for sd in data["states_def"]}
        transitions = {self._transition_from_dict(t, states) for t in data["transitions"]}
        is_root = True if data.get("is_root") else False
        return StateMachineDefinition(
            name=name,
            is_root=is_root,
            states_def=states,
            transitions=transitions,
            inner_end_state_to_outer_event=data.get("inner_end_state_to_outer_event") if not is_root else None
        )

    def state_machine_def_to_dict(self, state_machine_def: StateMachineDefinition) -> Dict[str, Any]:
        result = {
            "is_root": state_machine_def.is_root,
            "states_def": [self._state_def_to_dict(sd) for sd in
                           state_machine_def.states_def],
            "transitions": [self._transition_to_dict(t) for t in
                            state_machine_def.transitions],
        }
        if not state_machine_def.is_root and state_machine_def.inner_end_state_to_outer_event:
            result["inner_end_state_to_outer_event"] = state_machine_def.inner_end_state_to_outer_event
        return result

    @staticmethod
    def _state_def_to_dict(state_def: BaseStateDefinition) -> Dict[str, Any]:
        if isinstance(state_def, StateDefinition):
            actions_des = [{a.on_command: a.full_class_name} for a in state_def.actions_des]
            return {
                "name": state_def.name,
                "description": state_def.description,
                "state_context_class_name": state_def.state_context_class_name,
                "task_type": state_def.task_type.name,
                "actions": actions_des,
                "events": list(state_def.events),
                "is_start": state_def.is_start
            }
        elif isinstance(state_def, BaseStateDefinition):
            return {
                "name": state_def.name,
                "description": state_def.description,
                "state_context_class_name": state_def.state_context_class_name
            }

    def _state_def_from_dict(self, data: Dict[str, Any], state_machine_name: str,
                             exist_state_machine_names: Set) -> BaseStateDefinition:
        is_composite = self.is_composite_state(parent_state_machine_name=state_machine_name,
                                               state_name=data["name"],
                                               exist_state_machine_names=exist_state_machine_names)
        if data.get("actions"):
            return StateDefinition(
                name=data["name"],
                state_context_class_name=data["state_context_class_name"],
                description=data.get("description", ""),
                task_type=TaskType.get_type(data.get("task_type", "")),
                actions_des={ActionDescription(a) for a in data["actions"]},
                result_events=set(data["events"]),
                is_composite=is_composite,
                is_start=data.get("is_start")
            )
        else:
            return BaseStateDefinition(
                name=data["name"],
                description=data.get("description", ""),
                state_context_class_name=data.get("state_context_class_name")
            )

    @staticmethod
    def is_composite_state(parent_state_machine_name: str, state_name: str, exist_state_machine_names):
        state_machine_name = state_name
        if parent_state_machine_name:
            state_machine_name = f"{parent_state_machine_name}.{state_name}"
        return state_machine_name in exist_state_machine_names

    @classmethod
    def _transition_to_dict(self, transition: Transition) -> Dict[str, str]:
        return {
            "event": transition.event,
            "source": transition.source.name,
            "target": transition.target.name
        }

    @staticmethod
    def _transition_from_dict(data: Dict[str, str], states: Set[BaseStateDefinition]) -> Transition:
        source = None
        target = None

        # 查找 source
        for sd in states:
            if sd.name == data["source"]:
                source = sd
                break
        if source is None:
            raise ValueError(f"Source state {data['source']} not found")

        # 查找 target
        for sd in states:
            if sd.name == data["target"]:
                target = sd
                break
        if target is None:
            raise ValueError(f"Target state {data['target']} not found")

        return Transition(event=data["event"], source=source, target=target)


class StateContextBuilder:
    def __init__(self, state_machine_context_repository: "StateMachineRepository",
                 state_machine_definition_repository: "StateMachineDefinitionRepository"):
        self.state_machine_context_repository = state_machine_context_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    def build(self, state_def: BaseStateDefinition,
              id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        if type(state_def) is StateDefinition:
            state_def = cast(StateDefinition, state_def)
            full_class_name = state_def.state_context_class_name
            if not full_class_name:
                full_class_name = "thinker_ai.status_machine.state_machine.StateDefinition"
            if state_def.is_composite:
                return CompositeStateContext.from_class_name(id=id,
                                                             full_class_name=full_class_name,
                                                             state_def=state_def,
                                                             state_context_builder=self,
                                                             state_machine_repository=self.state_machine_context_repository,
                                                             state_machine_definition_repository=self.state_machine_definition_repository
                                                             )
            else:
                return StateContext.from_class_name(id=id,
                                                    full_class_name=full_class_name,
                                                    state_def=state_def)
        elif type(state_def) is BaseStateDefinition:
            full_class_name = state_def.state_context_class_name
            if not full_class_name:
                full_class_name = "thinker_ai.status_machine.state_machine.BaseStateContext"
            return BaseStateContext.from_class_name(id=id,
                                                    full_class_name=full_class_name,
                                                    state_def=state_def)


class StateMachine:
    def __init__(self, id: str,
                 state_machine_def_name: str,
                 current_state_context: BaseStateContext,
                 state_context_builder: StateContextBuilder,
                 state_machine_repository: "StateMachineRepository",
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 history: Optional[List[BaseStateContext]] = None):
        self.id = id
        self.state_context_builder = state_context_builder
        self.state_machine_def_name = state_machine_def_name
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
                self.state_machine_repository.set(self.id, self)
                return
        raise ValueError(
            f"No transition from state '{self.current_state_context.state_def.name}' with event '{event.name}'")

    def get_state_machine_def(self) -> StateMachineDefinition:
        return self.state_machine_definition_repository.get(self.state_machine_def_name)

    def last_state(self) -> Optional[StateContext]:
        return self.history[-1] if self.history else None

    def get_state_context(self, target: BaseStateDefinition) -> BaseStateContext:
        state_context = next((sc for sc in self.history if sc.state_def.name == target.name), None)
        if not state_context:
            state_context = self.state_context_builder.build(target)
        return state_context

    def to_outer_event(self, inner_event) -> Optional[Event]:
        state_machine_def = self.get_state_machine_def()
        if state_machine_def.inner_end_state_to_outer_event:
            return state_machine_def.to_outer_event(inner_event, self.current_state_context)


class StateMachineRepository(ABC):
    @abstractmethod
    def get(self, id: str) -> StateMachine:
        raise NotImplementedError

    @abstractmethod
    def set(self, id: str, state_machine_context: StateMachine):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError


class CompositeStateContext(StateContext):
    def __init__(self, id: str,
                 state_def: StateDefinition,
                 state_context_builder: StateContextBuilder,
                 state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, state_def)
        self.state_context_builder = state_context_builder
        self.state_machine_repository = state_machine_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    @classmethod
    def from_class_name(cls, full_class_name: str, **kwargs) -> 'CompositeStateContext':
        return from_class_name(cls, full_class_name, **kwargs)

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

    def get_state_def(self) -> StateDefinition:
        return cast(StateDefinition, self.state_def)

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, on_command: str) -> Optional[CompositeAction]:
        action = super().get_action(on_command)
        if isinstance(action, CompositeAction):
            return action
        else:
            raise TypeError("action not a CompositeAction")

    def get_state_machine(self) -> StateMachine:
        my_state_machine = self.state_machine_repository.get(self.id)
        if not my_state_machine:
            state_machine_definition = self.state_machine_definition_repository.get(
                self.get_state_def().name)
            my_state_machine = StateMachine(id=self.id,
                                            state_machine_def_name=self.get_state_def().name,
                                            current_state_context=self.state_context_builder.build(
                                                state_machine_definition.get_start_state_def()),
                                            state_context_builder=self.state_context_builder,
                                            state_machine_repository=self.state_machine_repository,
                                            state_machine_definition_repository=self.state_machine_definition_repository,
                                            history=[]
                                            )
            self.state_machine_repository.set(my_state_machine.id, my_state_machine)
        return my_state_machine

    def to_outer_event(self, inner_event: Event) -> Optional[Event]:
        if inner_event:
            return self.get_state_machine().to_outer_event(inner_event)


class ActionFactory:
    _registry = {}

    @classmethod
    def register_action(cls, action_class_name: str, action_cls: type):
        cls._registry[action_class_name] = action_cls

    @classmethod
    def create_action(cls, action_des: ActionDescription) -> Optional[Action]:
        action_cls: Action = cls._registry.get(action_des.full_class_name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for class '{action_des.full_class_name}'")
        return action_cls.from_class_name(on_command=action_des.on_command, full_class_name=action_des.full_class_name)
