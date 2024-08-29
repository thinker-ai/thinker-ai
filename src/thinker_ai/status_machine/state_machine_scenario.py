import json
import uuid
from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Tuple, cast, Any, Type

from thinker_ai.status_machine.base import from_class_name, ExecutorDescription
from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository, Event, StateDefinition, \
    BaseStateDefinition, Command, StateMachineDefinition, StateMachineDefinitionBuilder, CompositeStateDefinition


class ExecutorResult:
    def __init__(self, success: bool, result: Any = None, exception: Exception = None, event: Event = None):
        self.success = success
        self.result = result
        self.exception = exception
        self.event = event


class Executor(ABC):
    def __init__(self, on_command: str):
        self.on_command = on_command

    @abstractmethod
    def handle(self, command: Command, owner_state_scenario: "StateScenario", **kwargs) -> ExecutorResult:
        raise NotImplementedError

    @classmethod
    def from_class_name(cls, full_class_name: str, **kwargs) -> 'Executor':
        return from_class_name(cls, full_class_name, **kwargs)

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"


class CompositeExecutor(Executor):
    def __init__(self, on_command: str):
        super().__init__(on_command)

    @abstractmethod
    def handle(self, command: Command, owner_state_scenario: "CompositeStateScenario", **kwargs) -> ExecutorResult:
        # 子类实现该方法，分解为多个inner_command并根据需要调用该方法owner_state_scenario.handle_single_inner_command,最后通过to_outer_event返回结果
        raise NotImplementedError


class MockExecutor(Executor):

    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_scenario: "StateScenario", **kwargs) -> ExecutorResult:
        if command.name == self.on_command:
            result = ExecutorResult(success=True, event=Event(id=self.on_command, name=command.payload.get("event")))
            return result
        return ExecutorResult(success=False)


class MockCompositeExecutor(CompositeExecutor):
    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_scenario: "CompositeStateScenario", **kwargs) -> ExecutorResult:
        if command.name == self.on_command:
            inner_command_flows = owner_state_scenario.get_state_machine().get_state_machine_def().get_self_validate_commands_in_order()
            for inner_command_flow in inner_command_flows:
                for inner_command in inner_command_flow:
                    owner_state_scenario.handle_inner(inner_command)
            if owner_state_scenario.get_state_machine().current_state_scenario_des.state_def.is_terminal():
                result = ExecutorResult(success=True,event=Event(id=self.on_command, name=command.payload.get("event")))
                return result
        return ExecutorResult(success=False)


class ExecutorRegister:
    _registry = {}

    @classmethod
    def register_executor(cls, executor_class_name: str, executor_cls: type):
        cls._registry[executor_class_name] = executor_cls

    @classmethod
    def get_executor(cls, executor_des: ExecutorDescription) -> Optional[Executor]:
        executor_cls: Executor = cls._registry.get(executor_des.full_class_name)
        if executor_cls is None:
            raise ValueError(f"No Executor class registered for class '{executor_des.full_class_name}'")
        return executor_cls.from_class_name(on_command=executor_des.on_command,
                                            full_class_name=executor_des.full_class_name)


class BaseStateScenario:
    def __init__(self, scenario_id: str, state_def: BaseStateDefinition):
        self.scenario_id = scenario_id
        self.state_def = state_def

    @classmethod
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def get_state_def(self) -> BaseStateDefinition:
        return self.state_def

    def set_state_def(self, value: BaseStateDefinition):
        self.state_def = value


class StateScenario(BaseStateScenario):
    def __init__(self, scenario_id: str, state_def: StateDefinition):
        super().__init__(scenario_id, state_def)

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_executor(self, on_command: str) -> Optional[Executor]:
        for executor_des in self.state_def.executors_des:
            if executor_des.on_command == on_command:
                return ExecutorRegister.get_executor(executor_des)

    def assert_event(self, event: Event):
        if not event:
            raise RuntimeError(f'Event is None')
        found = False
        for event_def in self.state_def.events:
            if event.name == event_def.name:
                found = True
                break
        if not found:
            raise RuntimeError(f'Illegal event "{event.name}" for state {self.state_def.name}')

    def handle(self, command: Command, outer_state_machine: 'StateMachineScenario', **kwargs) -> ExecutorResult:
        if self.state_def.name != command.target:
            raise Exception(
                f"the current state scenario {self.state_def.name} do not the command target {command.target}")
        if command.payload and command.payload.get("self_validate"):
            executor = MockExecutor(command.name)
        else:
            executor = self.get_executor(command.name)
        if executor:
            result = executor.handle(command, self, **kwargs)
            if result.event:
                self.assert_event(result.event)
                outer_state_machine.on_event(result.event)
            return result
        else:
            raise Exception(
                f"Illegal command {command.name} for state {self.state_def.name}")


class StateScenarioBuilder(ABC):
    @staticmethod
    def build_state_scenario(state_def: StateDefinition, state_scenario_class_name: str,
                             scenario_id: Optional[str] = str(uuid.uuid4())) -> StateScenario:
        raise NotImplementedError

    @staticmethod
    def build_terminal_state_scenario(state_def: BaseStateDefinition, state_scenario_class_name: str,
                                      scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        raise NotImplementedError

    @classmethod
    def build_composite_state_scenario(cls,
                                       state_def: "CompositeStateDefinition",
                                       state_scenario_class_name: str,
                                       state_machine_scenario_repository: "StateMachineScenarioRepository",
                                       scenario_root_id: str,
                                       scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        raise NotImplementedError


class StateScenarioDescription:
    def __init__(self,
                 state_def: BaseStateDefinition,
                 state_scenario_builder_full_class_name: str,
                 state_machine_scenario_repository: "StateMachineScenarioRepository",
                 scenario_root_id: str,
                 scenario_id: Optional[str] = str(uuid.uuid4())
                 ):
        self.scenario_root_id: str = scenario_root_id
        self.scenario_id: str = scenario_id
        self.state_def = state_def
        self.state_scenario_builder_full_class_name = state_scenario_builder_full_class_name
        self.state_machine_repository = state_machine_scenario_repository

    def get_state_scenario_builder_class(self) -> "StateScenarioBuilder":
        if self.state_scenario_builder_full_class_name:
            state_scenario_builder_class = from_class_name(StateScenarioBuilder,
                                                           self.state_scenario_builder_full_class_name)
        else:
            state_scenario_builder_class = DefaultStateScenarioBuilder
        return state_scenario_builder_class

    def get_state_scenario(self, is_validate: bool) -> BaseStateScenario:
        if is_validate:
            # 模拟过程不用缓存，避免缓存中模拟的state_scenario替代并占位实际的state_scenario
            state_scenario = self._build_state_scenario(DefaultStateScenarioBuilder())
        else:
            state_scenario = StateContextRegister.get(self.scenario_id)
            if not state_scenario:
                state_scenario_builder = self.get_state_scenario_builder_class()
                state_scenario = self._build_state_scenario(state_scenario_builder)
                StateContextRegister.register(self.scenario_id, state_scenario)
        return state_scenario

    def _build_state_scenario(self, state_scenario_builder: StateScenarioBuilder):
        if isinstance(self.state_def, CompositeStateDefinition):
            state_scenario = (state_scenario_builder
                              .build_composite_state_scenario(state_def=self.state_def,
                                                              state_scenario_class_name=self.state_def.state_scenario_class_name,
                                                              state_machine_scenario_repository=self.state_machine_repository,
                                                              scenario_root_id=self.scenario_root_id,
                                                              scenario_id=self.scenario_id))
        elif isinstance(self.state_def, StateDefinition):
            state_scenario = (
                state_scenario_builder.build_state_scenario(state_def=self.state_def,
                                                            state_scenario_class_name=self.state_def.state_scenario_class_name,
                                                            scenario_id=self.scenario_id))
        else:
            state_scenario = (
                state_scenario_builder.build_terminal_state_scenario(state_def=self.state_def,
                                                                     state_scenario_class_name=self.state_def.state_scenario_class_name,
                                                                     scenario_id=self.scenario_id))
        return state_scenario


class DefaultStateScenarioBuilder(StateScenarioBuilder):
    @staticmethod
    def build_state_scenario(state_def: StateDefinition,
                             state_scenario_class_name: str,
                             scenario_id: Optional[str] = str(uuid.uuid4())) -> StateScenario:
        return StateScenario(scenario_id=scenario_id, state_def=state_def)

    @staticmethod
    def build_terminal_state_scenario(state_def: BaseStateDefinition,
                                      state_scenario_class_name: str,
                                      scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        return BaseStateScenario(scenario_id=scenario_id, state_def=state_def)

    @classmethod
    def build_composite_state_scenario(cls,
                                       state_def: "CompositeStateDefinition",
                                       state_scenario_class_name: str,
                                       state_machine_scenario_repository: "StateMachineScenarioRepository",
                                       scenario_root_id: str,
                                       scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        return CompositeStateScenario(scenario_id=scenario_id,
                                      scenario_root_id=scenario_root_id,
                                      state_def=state_def,
                                      state_scenario_builder_class=cls,
                                      state_machine_scenario_repository=state_machine_scenario_repository
                                      )


class StateMachineScenario:
    def __init__(self, scenario_id: str,
                 scenario_root_id: str,
                 state_machine_def: StateMachineDefinition,
                 current_state_scenario_des: StateScenarioDescription,
                 history: Optional[List[StateScenarioDescription]] = None):
        self.scenario_id = scenario_id
        self.scenario_root_id = scenario_root_id
        self.state_machine_def = state_machine_def
        self.current_state_scenario_des: StateScenarioDescription = current_state_scenario_des
        self.history: List[StateScenarioDescription] = history or []

    def handle(self, command: Command, **kwargs) -> ExecutorResult:
        is_validate = command.payload and command.payload.get("self_validate")
        current_state_scenario = self.current_state_scenario_des.get_state_scenario(is_validate)
        if isinstance(current_state_scenario, StateScenario):
            return current_state_scenario.handle(command, self, **kwargs)
        else:
            raise Exception(
                f"Terminal State {self.current_state_scenario_des.state_def.name} can not handle the command {command.name}!")

    def on_event(self, event: Event):
        # if not self.current_state_scenario_des.state_machine_definition_repository:
        #     raise ValueError("Repositories are not set")
        state_machine_definition = self.get_state_machine_def()
        transitions = state_machine_definition.transitions
        for transition in transitions:
            if transition.event == event.name and transition.source.name == self.current_state_scenario_des.state_def.name:
                self.history.append(self.current_state_scenario_des)
                self.current_state_scenario_des = self.creat_state_scenario_des(transition.target)
                self.current_state_scenario_des.state_machine_repository.set_scenario(self.scenario_root_id, self)
                return
        raise ValueError(
            f"No transition from state '{self.current_state_scenario_des.state_def.name}' with event '{event.name}'")

    def get_state_machine_def(self) -> StateMachineDefinition:
        return self.state_machine_def

    def last_state(self) -> Optional[StateScenario]:
        return self.history[-1] if self.history else None

    def creat_state_scenario_des(self, state_def: BaseStateDefinition) -> StateScenarioDescription:
        return StateScenarioDescription(
            state_scenario_builder_full_class_name=self.get_state_machine_def().state_scenario_builder_full_class_name,
            state_def=state_def,
            state_machine_scenario_repository=self.current_state_scenario_des.state_machine_repository,
            scenario_root_id=self.scenario_root_id,
            scenario_id=str(uuid.uuid4())
        )

    def to_outer_event(self, inner_event) -> Optional[Event]:
        state_machine_def = self.get_state_machine_def()
        if state_machine_def.inner_end_state_to_outer_event:
            return state_machine_def.to_outer_event(inner_event, self.current_state_scenario_des.state_def)

    def self_validate(self) -> List[Tuple[List[Command], bool]]:
        commands_orders = self.get_state_machine_def().get_self_validate_commands_in_order()
        result = []
        for commands_order in commands_orders:
            self.reset()
            for command in commands_order:
                self.handle(command)
            result.append((commands_order, self.current_state_scenario_des.state_def.is_terminal()))
        return result

    def reset(self):
        self.current_state_scenario_des = self.creat_state_scenario_des(
            self.get_state_machine_def().get_start_state_def())
        self.history.clear()


class StateMachineScenarioRepository(ABC):
    @abstractmethod
    def get(self, scenario_root_id: str, scenario_id: str) -> StateMachineScenario:
        raise NotImplementedError

    @abstractmethod
    def set_scenario(self, scenario_root_id: str, state_machine_scenario: StateMachineScenario):
        raise NotImplementedError

    @abstractmethod
    def set_dict(self, scenario_root_id: str, state_machine_dict: dict):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError


class CompositeStateScenario(StateScenario):
    def __init__(self,
                 scenario_id: str,
                 scenario_root_id: str,
                 state_def: CompositeStateDefinition,
                 state_scenario_builder_class: Type[StateScenarioBuilder],
                 state_machine_scenario_repository: StateMachineScenarioRepository
                 ):
        super().__init__(scenario_id, state_def)
        self.scenario_root_id = scenario_root_id
        self.state_scenario_builder_class = state_scenario_builder_class
        self.state_machine_scenario_repository = state_machine_scenario_repository
        self.state_machine = self.state_machine_scenario_repository.get(self.scenario_root_id, self.scenario_id)
        if not self.state_machine:
            self.state_machine = StateMachineScenarioBuilder.new_from_def(self.state_def.state_machine_definition,
                                                                            self.state_machine_scenario_repository)

    def handle(self, command: Command, outer_state_machine: 'StateMachineScenario', **kwargs) -> ExecutorResult:
        if self.state_def.name != command.target:
            raise Exception(
                f"the current state scenario {self.state_def.name} do not the command target {command.target}")
        if command.payload and command.payload.get("self_validate"):
            executor = MockCompositeExecutor(command.name)
        else:
            executor = self.get_executor(command.name)
        if executor:
            result = executor.handle(command, self, **kwargs)
            if result.event:
                self.assert_event(result.event)
                outer_state_machine.on_event(result.event)
            return result

    def handle_inner(self, inner_command: Command, **kwargs) -> ExecutorResult:
        inner_result = self.get_state_machine().handle(inner_command, **kwargs)
        inner_result.event = self.to_outer_event(inner_result.event)
        return inner_result

    def get_state_def(self) -> CompositeStateDefinition:
        return cast(CompositeStateDefinition, self.state_def)

    def set_state_def(self, value: CompositeStateDefinition):
        self.state_def = value

    def get_executor(self, on_command: str) -> Optional[CompositeExecutor]:
        executor = super().get_executor(on_command)
        if isinstance(executor, CompositeExecutor):
            return executor
        else:
            raise TypeError("executor not a CompositeExecutor")

    def get_state_machine(self) -> StateMachineScenario:
        return self.state_machine

    def to_outer_event(self, inner_event: Event) -> Optional[Event]:
        if inner_event:
            return self.get_state_machine().to_outer_event(inner_event)


class StateMachineScenarioBuilder:

    @classmethod
    def new_from_group_def_json(cls, state_machine_def_group_name: str,
                                state_machine_def_name: str,
                                def_json: str,
                                state_machine_definition_repository: StateMachineDefinitionRepository,
                                state_machine_scenario_repository: StateMachineScenarioRepository,
                                scenario_root_id: str = str(uuid.uuid4()),
                                ) -> StateMachineScenario:
        state_machine_def = StateMachineDefinitionBuilder.from_group_def_json(state_machine_def_group_name,
                                                                              state_machine_def_name, def_json,
                                                                              state_machine_definition_repository)
        state_machine = cls.new_from_def(state_machine_def,
                                         state_machine_scenario_repository, scenario_root_id)
        return state_machine

    @classmethod
    def new_from_def_name(cls, state_machine_def_group_name: str,
                          state_machine_def_name: str,
                          state_machine_definition_repository: StateMachineDefinitionRepository,
                          state_machine_scenario_repository: StateMachineScenarioRepository,
                          scenario_root_id: str = str(uuid.uuid4())) -> StateMachineScenario:
        state_machine_def = state_machine_definition_repository.get(state_machine_def_group_name,
                                                                    state_machine_def_name)
        state_machine = cls.new_from_def(state_machine_def,
                                         # state_machine_definition_repository,
                                         state_machine_scenario_repository, scenario_root_id)
        return state_machine

    @classmethod
    def new_from_def(cls, state_machine_def: StateMachineDefinition,
                     state_machine_scenario_repository: StateMachineScenarioRepository,
                     scenario_root_id: str = str(uuid.uuid4())) -> StateMachineScenario:
        start_state_scenario = StateScenarioDescription(
            state_scenario_builder_full_class_name=state_machine_def.state_scenario_builder_full_class_name,
            state_def=state_machine_def.get_start_state_def(),
            state_machine_scenario_repository=state_machine_scenario_repository,
            scenario_root_id=scenario_root_id,
            scenario_id=str(uuid.uuid4())
        )
        state_machine = StateMachineScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_root_id=scenario_root_id,
            state_machine_def=state_machine_def,
            current_state_scenario_des=start_state_scenario,
            history=[]
        )
        return state_machine

    @classmethod
    def state_machine_scenario_to_dict(cls, state_machine_scenario: StateMachineScenario) -> Dict[str, Any]:
        current_state_scenario = {
            "scenario_id": state_machine_scenario.current_state_scenario_des.scenario_id,
            "state_def_name": state_machine_scenario.current_state_scenario_des.state_def.name,
        }

        history_data = [
            {
                "scenario_id": scenario_des.scenario_id,
                "state_def_name": scenario_des.state_def.name,
            }
            for scenario_des in state_machine_scenario.history
        ]

        return {
            state_machine_scenario.scenario_id: {
                "state_machine_def_name": state_machine_scenario.get_state_machine_def().name,
                "state_machine_def_group_name": state_machine_scenario.get_state_machine_def().group_def_name,
                "current_state_scenario": current_state_scenario,
                "history": history_data
            }
        }

    @classmethod
    def state_machine_scenario_from_dict(cls, scenario_root_id: str, scenario_id: str, data: Dict[str, Any],
                                         state_machine_definition_repository: StateMachineDefinitionRepository,
                                         state_machine_scenario_repository: StateMachineScenarioRepository
                                         ) -> StateMachineScenario:
        state_machine_def = (state_machine_definition_repository
                             .get(data["state_machine_def_group_name"], data["state_machine_def_name"]))
        current_state_scenario_date = data["current_state_scenario"]
        current_state_def = next(
            sd for sd in state_machine_def.states_def if sd.name == current_state_scenario_date["state_def_name"]
        )
        current_state_scenario = StateScenarioDescription(
            state_scenario_builder_full_class_name=state_machine_def.state_scenario_builder_full_class_name,
            state_def=current_state_def,
            state_machine_scenario_repository=state_machine_scenario_repository,
            scenario_root_id=scenario_root_id,
            scenario_id=current_state_scenario_date["scenario_id"])

        history = [StateScenarioDescription(
            state_scenario_builder_full_class_name=state_machine_def.state_scenario_builder_full_class_name,
            state_def=next(sd for sd in state_machine_def.states_def if sd.name == one_history_data["state_def_name"]),
            state_machine_scenario_repository=state_machine_scenario_repository,
            scenario_root_id=scenario_root_id,
            scenario_id=one_history_data["scenario_id"],
        ) for one_history_data in data["history"]]

        state_machine = StateMachineScenario(
            scenario_id=scenario_id,
            scenario_root_id=scenario_root_id,
            state_machine_def=state_machine_def,
            current_state_scenario_des=current_state_scenario,
            history=history
        )
        return state_machine

    @classmethod
    def state_machine_scenario_to_json(cls, state_machine: StateMachineScenario) -> str:
        state_machine_dict = cls.state_machine_scenario_to_dict(state_machine)
        return json.dumps(state_machine_dict, indent=2, ensure_ascii=False)

    @classmethod
    def state_machine_scenario_from_json(cls, scenario_root_id: str, scenario_id: str,
                                         json_text: str,
                                         state_machine_definition_repository: StateMachineDefinitionRepository,
                                         state_machine_scenario_repository: StateMachineScenarioRepository
                                         ) -> "StateMachineScenario":
        data = json.loads(json_text)
        # 状态机的name和状态机json节点是kv关系，json_text只有状态机自身的信息，没有group_id和id的信息，所以，要把group_id和id独立传入，
        # 所以，TODO：key方式存储name,也许不是最好的方式，待改进
        return cls.state_machine_scenario_from_dict(scenario_root_id, scenario_id, data,
                                                    state_machine_definition_repository,
                                                    state_machine_scenario_repository)

    @classmethod
    def state_machine_group_from_json(cls, json_text: str,
                                      state_machine_definition_repository: StateMachineDefinitionRepository,
                                      state_machine_scenario_repository: StateMachineScenarioRepository
                                      ) -> dict[str, dict[str, StateMachineScenario]]:
        data = json.loads(json_text)
        return cls.state_machine_group_from_dict(data, state_machine_definition_repository,
                                                 state_machine_scenario_repository)

    @classmethod
    def state_machine_group_from_dict(cls, data: Dict[str, Any],
                                      state_machine_definition_repository: StateMachineDefinitionRepository,
                                      state_machine_scenario_repository: StateMachineScenarioRepository
                                      ) -> dict[str, dict[str, StateMachineScenario]]:
        result = {}
        for scenario_root_id in data.keys():
            group_data = data.get(scenario_root_id)
            root_scenario = {}
            for state_machine_id in group_data.keys():
                state_machine_data = group_data.get(state_machine_id)

                state_machine = StateMachineScenarioBuilder.state_machine_scenario_from_dict(scenario_root_id,
                                                                                             state_machine_id,
                                                                                             state_machine_data,
                                                                                             state_machine_definition_repository,
                                                                                             state_machine_scenario_repository)
                root_scenario[state_machine_id] = state_machine
            result[scenario_root_id] = root_scenario
        return result


class StateContextRegister:
    _registry = {}

    @classmethod
    def register(cls, full_class_name: str, state_scenario: StateScenario):
        cls._registry[full_class_name] = state_scenario

    @classmethod
    def get(cls, full_class_name: str) -> StateScenario:
        return cls._registry.get(full_class_name)
