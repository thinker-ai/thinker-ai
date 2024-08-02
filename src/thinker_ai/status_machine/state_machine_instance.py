import json
import uuid
from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Tuple, cast, Any, Type

from thinker_ai.status_machine.base import from_class_name, ActionDescription
from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository, Event, StateDefinition, \
    BaseStateDefinition, Command, StateMachineDefinition, StateMachineDefinitionBuilder


class ActionResult:
    def __init__(self, success: bool, result: Any = None, exception: Exception = None, event: Event = None):
        self.success = success
        self.result = result
        self.exception = exception
        self.event = event


class Action(ABC):
    def __init__(self, on_command: str):
        self.on_command = on_command

    @abstractmethod
    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
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
    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        # 子类实现该方法，分解为多个inner_command并根据需要调用该方法owner_state_context.handle_single_inner_command,最后通过to_outer_event返回结果
        raise NotImplementedError


class MockAction(Action):

    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            result = ActionResult(success=True, event=Event(id=self.on_command, name=command.payload.get("event")))
            return result
        return ActionResult(success=False)


class MockCompositeAction(CompositeAction):
    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_command_flows = (owner_state_context.get_state_machine()
                                   .get_state_machine_def().get_self_validate_commands_in_order())
            for inner_command_flow in inner_command_flows:
                for inner_command in inner_command_flow:
                    owner_state_context.handle_inner(inner_command)
            if owner_state_context.get_state_machine().current_state_context_des.state_def.is_terminal():
                result = ActionResult(success=True, event=Event(id=self.on_command, name=command.payload.get("event")))
                return result
        return ActionResult(success=False)


class ActionRegister:
    _registry = {}

    @classmethod
    def register_action(cls, action_class_name: str, action_cls: type):
        cls._registry[action_class_name] = action_cls

    @classmethod
    def get_action(cls, action_des: ActionDescription) -> Optional[Action]:
        action_cls: Action = cls._registry.get(action_des.full_class_name)
        if action_cls is None:
            raise ValueError(f"No Action class registered for class '{action_des.full_class_name}'")
        return action_cls.from_class_name(on_command=action_des.on_command, full_class_name=action_des.full_class_name)


class BaseStateContext:
    def __init__(self, instance_id: str, state_def: BaseStateDefinition):
        self.instance_id = instance_id
        self.state_def = state_def

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def get_state_def(self) -> BaseStateDefinition:
        return self.state_def

    def set_state_def(self, value: BaseStateDefinition):
        self.state_def = value


class StateContext(BaseStateContext):
    def __init__(self, instance_id: str, state_def: StateDefinition):
        super().__init__(instance_id, state_def)

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, on_command: str) -> Optional[Action]:
        for action_des in self.state_def.actions_des:
            if action_des.on_command == on_command:
                return ActionRegister.get_action(action_des)

    def assert_event(self, event: Event):
        if not event:
            raise RuntimeError(f'Event is None')
        if event.name not in self.state_def.events:
            raise RuntimeError(f'Illegal event "{event.name}" for state {self.state_def.name}')

    def handle(self, command: Command, outer_state_machine: 'StateMachine', **kwargs) -> ActionResult:
        if self.state_def.name != command.target:
            raise Exception(
                f"the current state context {self.state_def.name} do not the command target {command.target}")
        if command.payload and command.payload.get("self_validate"):
            action = MockAction(command.name)
        else:
            action = self.get_action(command.name)
        if action:
            result = action.handle(command, self, **kwargs)
            if result.event:
                self.assert_event(result.event)
                outer_state_machine.on_event(result.event)
            return result
        else:
            raise Exception(
                f"Illegal command {command.name} for state {self.state_def.name}")


class StateContextBuilder(ABC):
    @staticmethod
    def build_state_instance(state_def: StateDefinition, state_context_class_name: str,
                             instance_id: Optional[str] = str(uuid.uuid4())) -> StateContext:
        raise NotImplementedError

    @staticmethod
    def build_terminal_state_instance(state_def: BaseStateDefinition, state_context_class_name: str,
                                      instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        raise NotImplementedError

    @classmethod
    def build_composite_state_instance(cls, state_def: "StateDefinition", state_context_class_name: str,
                                       state_machine_definition_repository: "StateMachineDefinitionRepository",
                                       state_machine_context_repository: "StateMachineRepository",
                                       instance_group_id: str,
                                       instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        raise NotImplementedError


class StateContextDescription:
    def __init__(self,
                 state_def: BaseStateDefinition,
                 state_context_builder_full_class_name: str,
                 state_machine_definition_repository: "StateMachineDefinitionRepository",
                 state_machine_repository: "StateMachineRepository",
                 instance_group_id: str,
                 instance_id: Optional[str] = str(uuid.uuid4())
                 ):
        self.instance_group_id: str = instance_group_id
        self.instance_id: str = instance_id
        self.state_def = state_def
        self.state_context_builder_full_class_name = state_context_builder_full_class_name
        self.state_machine_definition_repository = state_machine_definition_repository
        self.state_machine_repository = state_machine_repository

    def get_state_context_builder_class(self) -> "StateContextBuilder":
        if self.state_context_builder_full_class_name:
            state_context_builder_class = from_class_name(StateContextBuilder,
                                                          self.state_context_builder_full_class_name)
        else:
            state_context_builder_class = DefaultStateContextBuilder
        return state_context_builder_class

    def get_state_context(self) -> BaseStateContext:
        state_context = StateContextRegister.get(self.instance_id)
        if not state_context:
            state_context_builder = self.get_state_context_builder_class()
            if isinstance(self.state_def, StateDefinition):
                if self.state_def.is_composite:
                    state_context = (state_context_builder
                                     .build_composite_state_instance(self.state_def,
                                                                     self.state_def.state_context_class_name,
                                                                     self.state_machine_definition_repository,
                                                                     self.state_machine_repository,
                                                                     self.instance_group_id,
                                                                     self.instance_id))
                else:
                    state_context = (
                        state_context_builder.build_state_instance(self.state_def,
                                                                   self.state_def.state_context_class_name,
                                                                   self.instance_id))
            else:
                state_context = (
                    state_context_builder.build_terminal_state_instance(self.state_def,
                                                                        self.state_def.state_context_class_name,
                                                                        self.instance_id))
            StateContextRegister.register(self.instance_id, state_context)
        return state_context


class DefaultStateContextBuilder(StateContextBuilder):
    @staticmethod
    def build_state_instance(state_def: StateDefinition,
                             state_context_class_name: str,
                             instance_id: Optional[str] = str(uuid.uuid4())) -> StateContext:
        return StateContext(instance_id=instance_id, state_def=state_def)

    @staticmethod
    def build_terminal_state_instance(state_def: BaseStateDefinition,
                                      state_context_class_name: str,
                                      instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        return BaseStateContext(instance_id=instance_id, state_def=state_def)

    @classmethod
    def build_composite_state_instance(cls, state_def: StateDefinition,
                                       state_context_class_name: str,
                                       state_machine_definition_repository: "StateMachineDefinitionRepository",
                                       state_machine_context_repository: "StateMachineRepository",
                                       instance_group_id: str,
                                       instance_id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        if state_def.is_composite:
            return CompositeStateContext(instance_id=instance_id,
                                         instance_group_id=instance_group_id,
                                         state_def=state_def,
                                         state_context_builder_class=cls,
                                         state_machine_context_repository=state_machine_context_repository,
                                         state_machine_definition_repository=state_machine_definition_repository
                                         )


class StateMachine:
    def __init__(self, instance_id: str,
                 instance_group_id: str,
                 state_machine_def_group_name: str,
                 state_machine_def_name: str,
                 current_state_context_des: StateContextDescription,
                 state_machine_repository: "StateMachineRepository",
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 history: Optional[List[StateContextDescription]] = None):
        self.instance_id = instance_id
        self.instance_group_id = instance_group_id
        self.state_machine_def_group_name = state_machine_def_group_name
        self.state_machine_def_name = state_machine_def_name
        self.current_state_context_des: StateContextDescription = current_state_context_des
        self.history: List[StateContextDescription] = history or []
        self.state_machine_repository: StateMachineRepository = state_machine_repository
        self.state_machine_definition_repository: StateMachineDefinitionRepository = state_machine_definition_repository

    def handle(self, command: Command, **kwargs) -> ActionResult:
        current_state_context = self.current_state_context_des.get_state_context()
        if isinstance(current_state_context, StateContext):
            return current_state_context.handle(command, self, **kwargs)
        else:
            raise Exception(
                f"Terminal State {self.current_state_context_des.state_def.name} can not handle the command {command.name}!")

    def on_event(self, event: Event):
        if not self.state_machine_definition_repository:
            raise ValueError("Repositories are not set")
        state_machine_definition = self.get_state_machine_def()
        transitions = state_machine_definition.transitions
        for transition in transitions:
            if transition.event == event.name and transition.source.name == self.current_state_context_des.state_def.name:
                self.history.append(self.current_state_context_des)
                self.current_state_context_des = self.creat_state_context_des(transition.target)
                self.state_machine_repository.set(self.instance_group_id, self)
                return
        raise ValueError(
            f"No transition from state '{self.current_state_context_des.state_def.name}' with event '{event.name}'")

    def get_state_machine_def(self) -> StateMachineDefinition:
        return self.state_machine_definition_repository.get(self.state_machine_def_group_name,
                                                            self.state_machine_def_name)

    def last_state(self) -> Optional[StateContext]:
        return self.history[-1] if self.history else None

    def creat_state_context_des(self, state_def: BaseStateDefinition) -> StateContextDescription:
        return StateContextDescription(
            state_context_builder_full_class_name=self.get_state_machine_def().state_context_builder_full_class_name,
            state_def=state_def,
            state_machine_definition_repository=self.state_machine_definition_repository,
            state_machine_repository=self.state_machine_repository,
            instance_group_id=self.instance_group_id,
            instance_id=str(uuid.uuid4())
        )

    def to_outer_event(self, inner_event) -> Optional[Event]:
        state_machine_def = self.get_state_machine_def()
        if state_machine_def.inner_end_state_to_outer_event:
            return state_machine_def.to_outer_event(inner_event, self.current_state_context_des.state_def)

    def self_validate(self) -> List[Tuple[List[Command], bool]]:
        commands_orders = self.get_state_machine_def().get_self_validate_commands_in_order()
        result = []
        for commands_order in commands_orders:
            self.reset()
            for command in commands_order:
                self.handle(command)
            result.append((commands_order, self.current_state_context_des.state_def.is_terminal()))
        return result

    def reset(self):
        self.current_state_context_des = self.creat_state_context_des(
            self.get_state_machine_def().get_start_state_def())
        self.history.clear()


class StateMachineRepository(ABC):
    @abstractmethod
    def get(self, instance_group_id: str, instance_id: str) -> StateMachine:
        raise NotImplementedError

    @abstractmethod
    def set(self, instance_group_id: str, state_machine_instance: StateMachine):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError


class CompositeStateContext(StateContext):
    def __init__(self, instance_id: str, instance_group_id: str,
                 state_def: StateDefinition,
                 state_context_builder_class: Type[StateContextBuilder],
                 state_machine_context_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(instance_id, state_def)
        self.instance_group_id = instance_group_id
        self.state_context_builder_class = state_context_builder_class
        self.state_machine_repository = state_machine_context_repository
        self.state_machine_definition_repository = state_machine_definition_repository

    def handle(self, command: Command, outer_state_machine: 'StateMachine', **kwargs) -> ActionResult:
        if self.state_def.name != command.target:
            raise Exception(
                f"the current state context {self.state_def.name} do not the command target {command.target}")
        if command.payload and command.payload.get("self_validate"):
            action = MockCompositeAction(command.name)
        else:
            action = self.get_action(command.name)
        if action:
            result = action.handle(command, self, **kwargs)
            if result.event:
                self.assert_event(result.event)
                outer_state_machine.on_event(result.event)
            return result

    def handle_inner(self, inner_command: Command, **kwargs) -> ActionResult:
        inner_result = self.get_state_machine().handle(inner_command, **kwargs)
        inner_result.event = self.to_outer_event(inner_result.event)
        return inner_result

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
        result_state_machine = self.state_machine_repository.get(self.instance_group_id, self.instance_id)
        if not result_state_machine:
            state_machine_definition = self.state_machine_definition_repository.get(
                self.get_state_def().group_def_name, self.get_state_def().name)
            start_state_context_description = StateContextDescription(
                state_context_builder_full_class_name=state_machine_definition.state_context_builder_full_class_name,
                state_def=state_machine_definition.get_start_state_def(),
                state_machine_definition_repository=self.state_machine_definition_repository,
                state_machine_repository=self.state_machine_repository,
                instance_group_id=self.instance_group_id,
                instance_id=str(uuid.uuid4())
            )
            result_state_machine = StateMachine(instance_id=self.instance_id,
                                                instance_group_id=self.instance_group_id,
                                                state_machine_def_name=state_machine_definition.name,
                                                state_machine_def_group_name=state_machine_definition.group_def_name,
                                                current_state_context_des=start_state_context_description,
                                                state_machine_repository=self.state_machine_repository,
                                                state_machine_definition_repository=self.state_machine_definition_repository,
                                                history=[]
                                                )
            self.state_machine_repository.set(self.instance_group_id, result_state_machine)
        return result_state_machine

    def to_outer_event(self, inner_event: Event) -> Optional[Event]:
        if inner_event:
            return self.get_state_machine().to_outer_event(inner_event)


class StateMachineInstanceBuilder:

    @classmethod
    def new_from_json_def_json(cls, state_machine_def_group_name: str,
                               state_machine_def_name: str,
                               def_json:str,
                               state_machine_definition_repository: StateMachineDefinitionRepository,
                               state_machine_context_repository: StateMachineRepository,
                               instance_group_id: str = str(uuid.uuid4())) -> StateMachine:
        state_machine_def = StateMachineDefinitionBuilder.from_group_def_json(state_machine_def_group_name, state_machine_def_name, def_json)
        state_machine = cls.new_from_def(state_machine_def,state_machine_definition_repository,state_machine_context_repository,instance_group_id)
        return state_machine
    @classmethod
    def new_from_def_name(cls, state_machine_def_group_name: str,
                          state_machine_def_name: str,
                          state_machine_definition_repository: StateMachineDefinitionRepository,
                          state_machine_context_repository: StateMachineRepository,
                          instance_group_id: str = str(uuid.uuid4())) -> StateMachine:

        state_machine_def = (state_machine_definition_repository
                             .get(state_machine_def_group_name, state_machine_def_name))
        state_machine = cls.new_from_def(state_machine_def,state_machine_definition_repository,state_machine_context_repository,instance_group_id)
        return state_machine

    @classmethod
    def new_from_def(cls, state_machine_def: StateMachineDefinition,
                     state_machine_definition_repository: StateMachineDefinitionRepository,
                     state_machine_context_repository: StateMachineRepository,
                     instance_group_id: str = str(uuid.uuid4())) -> StateMachine:
        start_state_context = StateContextDescription(
            state_context_builder_full_class_name=state_machine_def.state_context_builder_full_class_name,
            state_def=state_machine_def.get_start_state_def(),
            state_machine_definition_repository=state_machine_definition_repository,
            state_machine_repository=state_machine_context_repository,
            instance_group_id=instance_group_id,
            instance_id=str(uuid.uuid4())
        )
        state_machine = StateMachine(
            instance_id=str(uuid.uuid4()),
            instance_group_id=instance_group_id,
            state_machine_def_group_name=state_machine_def.group_def_name,
            state_machine_def_name=state_machine_def.name,
            current_state_context_des=start_state_context,
            history=[],
            state_machine_repository=state_machine_context_repository,
            state_machine_definition_repository=state_machine_definition_repository
        )
        return state_machine

    @classmethod
    def state_machine_to_dict(cls, state_machine: StateMachine) -> Dict[str, Any]:
        current_state_context = {
            "instance_id": state_machine.current_state_context_des.instance_id,
            "state_def_name": state_machine.current_state_context_des.state_def.name,
        }

        history_data = [
            {
                "instance_id": context_des.instance_id,
                "state_def_name": context_des.state_def.name,
            }
            for context_des in state_machine.history
        ]

        return {
            "state_machine_def_name": state_machine.state_machine_def_name,
            "state_machine_def_group_name": state_machine.get_state_machine_def().group_def_name,
            "current_state_context": current_state_context,
            "history": history_data
        }

    @classmethod
    def state_machine_from_dict(cls, instance_group_id: str, instance_id: str, data: Dict[str, Any],
                                state_machine_definition_repository: StateMachineDefinitionRepository,
                                state_machine_context_repository: StateMachineRepository
                                ) -> StateMachine:
        state_machine_def = (state_machine_definition_repository
                             .get(data["state_machine_def_group_name"], data["state_machine_def_name"]))
        current_state_context_date = data["current_state_context"]
        current_state_def = next(
            sd for sd in state_machine_def.states_def if sd.name == current_state_context_date["state_def_name"]
        )
        current_state_context = StateContextDescription(
            state_context_builder_full_class_name=state_machine_def.state_context_builder_full_class_name,
            state_def=current_state_def,
            state_machine_definition_repository=state_machine_definition_repository,
            state_machine_repository=state_machine_context_repository,
            instance_group_id=instance_group_id,
            instance_id=current_state_context_date["instance_id"])

        history = [StateContextDescription(
            state_context_builder_full_class_name=state_machine_def.state_context_builder_full_class_name,
            state_def=next(sd for sd in state_machine_def.states_def if sd.name == context_data["state_def_name"]),
            state_machine_definition_repository=state_machine_definition_repository,
            state_machine_repository=state_machine_context_repository,
            instance_group_id=instance_group_id,
            instance_id=context_data["instance_id"],
        ) for context_data in data["history"]]

        state_machine = StateMachine(
            instance_id=instance_id,
            instance_group_id=instance_group_id,
            state_machine_def_group_name=data["state_machine_def_group_name"],
            state_machine_def_name=data["state_machine_def_name"],
            current_state_context_des=current_state_context,
            history=history,
            state_machine_repository=state_machine_context_repository,
            state_machine_definition_repository=state_machine_definition_repository
        )
        return state_machine

    @classmethod
    def state_machine_to_json(cls, state_machine: StateMachine) -> str:
        state_machine_dict = cls.state_machine_to_dict(state_machine)
        return json.dumps(state_machine_dict, indent=2, ensure_ascii=False)

    @classmethod
    def state_machine_from_json(cls, instance_group_id: str, instance_id: str,
                                json_text: str,
                                state_machine_definition_repository: StateMachineDefinitionRepository,
                                state_machine_context_repository: StateMachineRepository
                                ) -> "StateMachine":
        data = json.loads(json_text)
        # 状态机的name和状态机json节点是kv关系，json_text只有状态机自身的信息，没有group_id和id的信息，所以，要把group_id和id独立传入，
        # 所以，TODO：key方式存储name,也许不是最好的方式，待改进
        return cls.state_machine_from_dict(instance_group_id, instance_id, data, state_machine_definition_repository,
                                           state_machine_context_repository)

    @classmethod
    def state_machine_group_from_json(cls, json_text: str,
                                      state_machine_definition_repository: StateMachineDefinitionRepository,
                                      state_machine_context_repository: StateMachineRepository
                                      ) -> dict[str, dict[str, StateMachine]]:
        data = json.loads(json_text)
        return cls.state_machine_group_from_dict(data, state_machine_definition_repository,
                                                 state_machine_context_repository)

    @classmethod
    def state_machine_group_from_dict(cls, data: Dict[str, Any],
                                      state_machine_definition_repository: StateMachineDefinitionRepository,
                                      state_machine_context_repository: StateMachineRepository
                                      ) -> dict[str, dict[str, StateMachine]]:
        result = {}
        for instance_group_id in data.keys():
            group_data = data.get(instance_group_id)
            root_instance = {}
            for state_machine_id in group_data.keys():
                state_machine_data = group_data.get(state_machine_id)

                state_machine = StateMachineInstanceBuilder.state_machine_from_dict(instance_group_id,
                                                                                    state_machine_id,
                                                                                    state_machine_data,
                                                                                    state_machine_definition_repository,
                                                                                    state_machine_context_repository)
                root_instance[state_machine_id] = state_machine
            result[instance_group_id] = root_instance
        return result


class StateContextRegister:
    _registry = {}

    @classmethod
    def register(cls, full_class_name: str, state_context: StateContext):
        cls._registry[full_class_name] = state_context

    @classmethod
    def get(cls, full_class_name: str) -> StateContext:
        return cls._registry.get(full_class_name)
