import importlib
import json
import uuid
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, Set, cast, TypeVar, Type, Tuple

from thinker_ai.status_machine.task_desc import TaskTypeDef, TaskType

T = TypeVar('T')


def get_class_from_full_class_name(full_class_name: str):
    if '.' in full_class_name:
        module_name, class_name = full_class_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        context_class = getattr(module, class_name)
    else:
        # 假设类名在当前命名空间中
        context_class = globals().get(full_class_name)
        if context_class is None:
            raise ValueError(f"Class '{full_class_name}' not found in the current namespace.")

    return context_class


def from_class_name(cls: Type[T], full_class_name: str, **kwargs: Any) -> T:
    context_class = get_class_from_full_class_name(full_class_name)
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


class ActionDescription:
    def __init__(self, data: dict):
        self.on_command = data["on_command"]
        if data.get("pre_check_list"):
            self.pre_check_list: list[str] = [item for item in data.get("pre_check_list")]
        self.full_class_name = data["full_class_name"]
        if data.get("post_check_list"):
            self.post_check_list: list[str] = [item for item in data.get("post_check_list")]


class BaseStateDefinition:
    def __init__(self,
                 name: str,
                 description: str,
                 state_context_class_name: str
                 ):
        self.name = name
        self.description = description
        self.state_context_class_name = state_context_class_name

    def is_terminal(self):
        return type(self) is BaseStateDefinition


class StateDefinition(BaseStateDefinition):
    def __init__(self, name: str, group_name: str, description: str, task_type: TaskTypeDef,
                 state_context_class_name: str,
                 actions_des: Set[ActionDescription], result_events: Set[str], is_start: bool = False,
                 is_composite: bool = False):
        super().__init__(name, description, state_context_class_name)
        self.group_name = group_name
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
    def get_full_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def get_state_def(self) -> BaseStateDefinition:
        return self.state_def

    def set_state_def(self, value: BaseStateDefinition):
        self.state_def = value


class StateContext(BaseStateContext):
    def __init__(self, id: str, state_def: StateDefinition):
        super().__init__(id, state_def)

    def get_state_def(self) -> StateDefinition:
        return self.state_def

    def set_state_def(self, value: StateDefinition):
        self.state_def = value

    def get_action(self, on_command: str) -> Optional[Action]:
        action_des = self.get_action_des(on_command)
        if action_des:
            return ActionRegister.create_action(action_des)
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


class Transition:
    def __init__(self, event: str, source: BaseStateDefinition, target: BaseStateDefinition):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self,
                 group_name: str,
                 name: str,
                 states_def: Set[BaseStateDefinition],
                 transitions: Set[Transition],
                 state_context_builder_full_class_name: Optional[str],
                 is_root: bool = False,
                 inner_end_state_to_outer_event: Optional[Dict[str, str]] = None):
        self.name = name
        self.state_context_builder_full_class_name = state_context_builder_full_class_name
        self.group_name = group_name
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

    def get_start_state_def(self) -> StateDefinition:
        for state in self.states_def:
            if isinstance(state, StateDefinition) and state.is_start:
                return state
        raise Exception("Start state not found")

    def to_outer_event(self, inner_event: Event, state_def: BaseStateDefinition) -> Optional[Event]:
        if type(state_def) is not BaseStateDefinition:
            return None
        outer_event_name = self.inner_end_state_to_outer_event.get(state_def.name)
        if outer_event_name is None:
            return None
        return Event(id=inner_event.id, name=outer_event_name, payload=inner_event.payload)

    def get_state_machine_create_order(self, state_machine_def_repo: "StateMachineDefinitionRepository",
                                       state_machine_def_group: str = None,
                                       state_machine_name: str = None,
                                       sorted_state_defs: Optional[List[Tuple[str, StateDefinition]]] = None) \
            -> List[Tuple[str, StateDefinition]]:
        if not state_machine_def_group:
            state_machine_def_group = self.group_name
            state_machine_name = self.name
            state_machine_def = self
        else:
            state_machine_def = state_machine_def_repo.get(state_machine_def_group, state_machine_name)
            if not state_machine_def:
                return sorted_state_defs

        if sorted_state_defs is None:
            sorted_state_defs = []
        visited: Set[str] = set()
        stack: List[Tuple[str, BaseStateDefinition]] = []

        if not state_machine_def:
            return sorted_state_defs

        def visit(state: BaseStateDefinition):
            state_def_name = f"{state_machine_def.name}.{state.name}"
            if state_def_name in visited:
                return
            visited.add(state_def_name)
            if (isinstance(state, StateDefinition)
                    and state.task_type
                    and state.task_type.name == TaskType.STATE_MACHINE_PLAN.type_name
                    and state_def_name not in state_machine_def_repo.get_state_machine_names(self.group_name)
            ):
                stack.append((state_def_name, state))
            for transition in state_machine_def.transitions:
                if transition.source.name == state.name:
                    visit(transition.target)

            # Process sub-state machine if any
            if isinstance(state, StateDefinition) and state.is_composite:
                self.get_state_machine_create_order(state_machine_def_repo=state_machine_def_repo,
                                                    state_machine_def_group=state_machine_def_group,
                                                    state_machine_name=f"{state_machine_name}.{state.name}",
                                                    sorted_state_defs=sorted_state_defs)

        start_state = state_machine_def.get_start_state_def()
        if start_state:
            visit(start_state)

        sorted_state_defs.extend(stack)  # Reverse the stack to get the correct order

        return sorted_state_defs

    def get_self_validate_commands_in_order(self) -> List[List[Command]]:
        paths = self._get_state_validate_paths()
        command_orders = []
        for path in paths:
            command_order = []
            for i in range(len(path) - 1):
                state_def_name, state = path[i]
                next_state_def_name, next_state = path[i + 1]

                # Find the transition that matches this state to the next state
                transition = self.get_transition(state.name, next_state.name)
                if transition:
                    # Find the action associated with this transition
                    if isinstance(state, StateDefinition):
                        action = self._from_event_to_mock_action(state, transition.event)
                        if action:
                            command = Command(
                                name=action.on_command,
                                target=transition.source.name,
                                payload={
                                    "self_validate": True,
                                    "event": transition.event
                                }
                            )
                            command_order.append(command)
            command_orders.append(command_order)
        return command_orders

    def get_transition(self, source_state_name: str, target_state_name: str) -> Optional[Transition]:
        for transition in self.transitions:
            if transition.source.name == source_state_name and transition.target.name == target_state_name:
                return transition
        return None

    @staticmethod
    def _from_event_to_mock_action(state: StateDefinition, event: str) -> Optional[
        ActionDescription]:
        for action in state.actions_des:
            if f"{action.on_command}_handled" == event:  # Assuming event name is used as command in mock action
                return action
        return None

    def _get_state_validate_paths(self) -> List[List[Tuple[str, BaseStateDefinition]]]:
        class Branch:
            def __init__(self, from_state: str):
                self.from_state = from_state
                self.to = {}  # Dictionary to store target states and their visited status

            def add_to_state(self, to_state: str):
                self.to[to_state] = False  # Initially, set the visited status to False

            def mark_visited(self, to_state: str):
                if to_state in self.to:
                    self.to[to_state] = True

        def find_all_branches() -> dict:
            branches = {}

            for state_def in self.states_def:
                from_state_name = state_def.name
                if from_state_name not in branches:
                    branch = Branch(from_state_name)
                    for transition in self.transitions:
                        if transition.source.name == from_state_name:
                            branch.add_to_state(transition.target.name)
                    if len(branch.to.items()) > 1:  # 否则不叫分支
                        branches[from_state_name] = branch
            return branches

        # Initialize branches
        branches = find_all_branches()

        def has_unvisited_branch() -> bool:
            for branch in branches.values():
                for to_state, visited in branch.to.items():
                    if not visited:
                        return True
            return False

        def merger_paths(execution_plan: List[List[Tuple[str, BaseStateDefinition]]]) -> List[
            List[Tuple[str, BaseStateDefinition]]]:
            terminal_paths = []
            un_terminal_paths = []
            for plan in execution_plan:
                _, state = plan[-1]
                if state.is_terminal():
                    terminal_paths.append(plan)
                else:
                    un_terminal_paths.append(plan)
            result_paths = []
            result_paths.extend(terminal_paths)
            for path_a in un_terminal_paths:
                for path_b in terminal_paths:
                    tail_name, _ = path_a[-1]
                    # 找到 path_b 中最后一个与 tail_name 匹配的元素
                    last_match_idx = -1
                    for idx, (name, _) in enumerate(path_b):
                        if name == tail_name:
                            last_match_idx = idx
                    if last_match_idx != -1:
                        # 合并 path_a 和 path_b，从 path_b 的 last_match_idx 位置开始连接到 path_a 的末尾
                        merged_path = path_a + path_b[last_match_idx + 1:]
                        result_paths.append(merged_path)

            return result_paths

        execution_plan = []
        while True:
            current_path = []

            def visit(state: BaseStateDefinition) -> bool:
                state_def_name = f"{self.name}.{state.name}"
                current_path.append((state_def_name, state))
                if state.is_terminal():
                    return True  # 表示退出上层的for循环
                for transition in self.transitions:
                    if transition.source.name == state.name:
                        next_state = transition.target
                        branch: Branch = branches.get(transition.source.name)
                        if branch:  # 存在分支
                            visited = branch.to.get(next_state.name)
                            if visited:  # 且访问过
                                continue
                            else:
                                branch.mark_visited(next_state.name)
                        return visit(next_state)  # 表示退出for循环

            start_state = self.get_start_state_def()
            if start_state:
                visit(start_state)
                execution_plan.append(current_path)

            # Check if there are any unvisited branches left
            if not has_unvisited_branch():
                break

        return merger_paths(execution_plan)


class StateMachineDefinitionRepository(ABC):
    @abstractmethod
    def get(self, state_machine_def_group_name: str, state_machine_name: str) -> StateMachineDefinition:
        raise NotImplementedError

    @abstractmethod
    def set(self, state_machine_def_group_name: str, state_machine_def: StateMachineDefinition):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError

    @abstractmethod
    def get_root_state_machine_name(self, state_machine_def_group_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_state_machine_names(self, state_machine_def_group_name: str):
        raise NotImplementedError


class StateMachineDefinitionBuilder:

    def root_from_dict(self, group_name: str, state_machine_defs: dict) -> StateMachineDefinition:
        for name, state_machine_def in iter(state_machine_defs.items()):
            if state_machine_def.get("is_root"):
                return self.state_machine_def_from_dict(group_name, name, state_machine_def,
                                                        set(state_machine_defs.keys()))

    def root_to_dict(self, root_state_machine_def: StateMachineDefinition) -> dict:
        if root_state_machine_def.is_root:
            return {root_state_machine_def.name: self.state_machine_def_to_dict(root_state_machine_def)}

    def root_from_json(self, group_name: str, json_data: str) -> StateMachineDefinition:
        data = json.loads(json_data)
        group_data = data.get(group_name)
        if group_data:
            return self.root_from_dict(group_name, group_data)

    def root_to_json(self, root_state_machine_def: StateMachineDefinition) -> str:
        data = self.root_to_dict(root_state_machine_def)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def state_machine_def_from_dict(self, group_name: str, state_machine_name: str, data: Dict[str, Any],
                                    exist_state_machine_names: Set) -> StateMachineDefinition:
        states = {self._state_def_from_dict(data=sd,
                                            group_name=group_name,
                                            state_machine_name=state_machine_name,
                                            exist_state_machine_names=exist_state_machine_names)
                  for sd in data["states_def"]}
        transitions = {self._transition_from_dict(t, states) for t in data["transitions"]}
        is_root = True if data.get("is_root") else False
        return StateMachineDefinition(
            name=state_machine_name,
            group_name=group_name,
            is_root=is_root,
            state_context_builder_full_class_name=data.get("state_context_builder_full_class_name"),
            states_def=states,
            transitions=transitions,
            inner_end_state_to_outer_event=data.get("inner_end_state_to_outer_event") if not is_root else None
        )

    def state_machine_def_to_dict(self, state_machine_def: StateMachineDefinition) -> Dict[str, Any]:
        result = {
            "is_root": state_machine_def.is_root,
            "state_context_builder_full_class_name": state_machine_def.state_context_builder_full_class_name,
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
            actions_des = [{"on_command": a.on_command,
                            "full_class_name": a.full_class_name,
                            "pre_check_list": a.pre_check_list,
                            "post_check_list": a.post_check_list
                            } for a in state_def.actions_des]
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

    def _state_def_from_dict(self, data: Dict[str, Any],
                             group_name: str,
                             state_machine_name: str,
                             exist_state_machine_names: Set) -> BaseStateDefinition:
        is_composite = self.is_composite_state(parent_state_machine_name=state_machine_name,
                                               state_name=data["name"],
                                               exist_state_machine_names=exist_state_machine_names)
        if data.get("actions"):
            return StateDefinition(
                name=data["name"],
                group_name=group_name,
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


class StateContextDescription:
    def __init__(self,
                 state_def: BaseStateDefinition,
                 state_context_builder_full_class_name: str,
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 state_machine_repository: "StateMachineRepository",
                 instance_id: Optional[str] = str(uuid.uuid4())
                 ):
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
                                                                     self.instance_id))
                else:
                    state_context = (
                        state_context_builder.build_state_instance(self.state_def,
                                                                   self.state_def.state_context_class_name,
                                                                   self.instance_id))
            else:
                state_context = (
                    state_context_builder.build_end_state_instance(self.state_def,
                                                                   self.state_def.state_context_class_name,
                                                                   self.instance_id))
            StateContextRegister.register(self.instance_id, state_context)
        return state_context


class StateContextBuilder(ABC):
    @staticmethod
    def build_state_instance(state_def: StateDefinition, state_context_class_name: str,
                             id: Optional[str] = str(uuid.uuid4())) -> StateContext:
        raise NotImplementedError

    @staticmethod
    def build_end_state_instance(state_def: BaseStateDefinition, state_context_class_name: str,
                                 id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        raise NotImplementedError

    @classmethod
    def build_composite_state_instance(cls, state_def: StateDefinition, state_context_class_name: str,
                                       state_machine_definition_repository: "StateMachineDefinitionRepository",
                                       state_machine_context_repository: "StateMachineRepository",
                                       id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        raise NotImplementedError


class DefaultStateContextBuilder(StateContextBuilder):
    @staticmethod
    def build_state_instance(state_def: StateDefinition,
                             state_context_class_name: str,
                             id: Optional[str] = str(uuid.uuid4())) -> StateContext:
        return StateContext(id=id, state_def=state_def)

    @staticmethod
    def build_end_state_instance(state_def: BaseStateDefinition,
                                 state_context_class_name: str,
                                 id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        return BaseStateContext(id=id, state_def=state_def)

    @classmethod
    def build_composite_state_instance(cls, state_def: StateDefinition,
                                       state_context_class_name: str,
                                       state_machine_definition_repository: "StateMachineDefinitionRepository",
                                       state_machine_context_repository: "StateMachineRepository",
                                       id: Optional[str] = str(uuid.uuid4())) -> BaseStateContext:
        if state_def.is_composite:
            return CompositeStateContext(id=id,
                                         state_def=state_def,
                                         state_context_builder_class=cls,
                                         state_machine_repository=state_machine_context_repository,
                                         state_machine_definition_repository=state_machine_definition_repository
                                         )


class StateMachine:
    def __init__(self, id: str,
                 state_machine_def_group_name: str,
                 state_machine_def_name: str,
                 current_state_context_des: StateContextDescription,
                 state_machine_repository: "StateMachineRepository",
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 history: Optional[List[StateContextDescription]] = None):
        self.id = id
        self.state_machine_def_group_name = state_machine_def_group_name
        self.state_machine_def_name = state_machine_def_name
        self.current_state_context_des: StateContextDescription = current_state_context_des
        self.history: List[BaseStateContext] = history or []
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
        current_state_context = self.current_state_context_des.get_state_context()
        for transition in transitions:
            if transition.event == event.name and transition.source.name == self.current_state_context_des.state_def.name:
                self.history.append(current_state_context)
                self.current_state_context_des = self.creat_state_context_des(transition.target)
                self.state_machine_repository.set(self)
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
    def get(self, instance_id: str) -> StateMachine:
        raise NotImplementedError

    @abstractmethod
    def set(self, state_machine_instance: StateMachine):
        raise NotImplementedError

    @abstractmethod
    def to_file(self, base_dir: str, file_name: str):
        raise NotImplementedError


class CompositeStateContext(StateContext):
    def __init__(self, id: str,
                 state_def: StateDefinition,
                 state_context_builder_class: StateContextBuilder.__class__,
                 state_machine_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(id, state_def)
        self.state_context_builder_class = state_context_builder_class
        self.state_machine_repository = state_machine_repository
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
        my_state_machine = self.state_machine_repository.get(self.id)
        if not my_state_machine:
            state_machine_definition = self.state_machine_definition_repository.get(
                self.get_state_def().group_name, self.get_state_def().name)
            my_state_machine = StateMachine(id=self.id,
                                            state_machine_def_name=state_machine_definition.name,
                                            state_machine_def_group_name=state_machine_definition.group_name,
                                            current_state_context_des=self.state_context_builder_class.build(
                                                state_machine_definition.get_start_state_def(),
                                                self.state_machine_definition_repository,
                                                self.state_machine_repository
                                            ),
                                            state_machine_repository=self.state_machine_repository,
                                            state_machine_definition_repository=self.state_machine_definition_repository,
                                            history=[]
                                            )
            self.state_machine_repository.set(my_state_machine)
        return my_state_machine

    def to_outer_event(self, inner_event: Event) -> Optional[Event]:
        if inner_event:
            return self.get_state_machine().to_outer_event(inner_event)


class StateMachineBuilder:
    @staticmethod
    def new(state_machine_def_group_name: str,
            state_machine_def_name: str,
            state_machine_definition_repository: StateMachineDefinitionRepository,
            state_machine_context_repository: StateMachineRepository
            ) -> StateMachine:
        state_machine_def = (state_machine_definition_repository
                             .get(state_machine_def_group_name, state_machine_def_name))
        start_state_def = state_machine_def.get_start_state_def()
        start_state_context = StateContextDescription(
            state_context_builder_full_class_name=state_machine_def.state_context_builder_full_class_name,
            state_def=start_state_def,
            state_machine_definition_repository=state_machine_definition_repository,
            state_machine_repository=state_machine_context_repository,
            instance_id=str(uuid.uuid4())
        )
        state_machine = StateMachine(
            id=str(uuid.uuid4()),
            state_machine_def_group_name=state_machine_def_group_name,
            state_machine_def_name=state_machine_def.name,
            current_state_context_des=start_state_context,
            history=[],
            state_machine_repository=state_machine_context_repository,
            state_machine_definition_repository=state_machine_definition_repository
        )
        return state_machine

    def to_json(self, state_machine: StateMachine) -> str:
        state_machine_dict = self.to_dict(state_machine)
        return json.dumps(state_machine_dict, indent=2, ensure_ascii=False)

    def from_json(self, id: str, json_text: str,
                  state_machine_definition_repository: StateMachineDefinitionRepository,
                  state_machine_context_repository: StateMachineRepository
                  ) -> StateMachine:
        state_machine_dict = json.loads(json_text)
        return self.from_dict(id,
                              state_machine_dict,
                              state_machine_definition_repository,
                              state_machine_context_repository)

    @staticmethod
    def to_dict(state_machine: StateMachine) -> Dict[str, Any]:
        current_state_context = {
            "id": state_machine.current_state_context_des.instance_id,
            "state_def_name": state_machine.current_state_context_des.state_def.name,
        }

        history_data = [
            {
                "id": context.id,
                "state_def_name": context.state_def.name,
            }
            for context in state_machine.history
        ]

        return {
            "state_machine_def_name": state_machine.state_machine_def_name,
            "state_machine_def_group_name": state_machine.get_state_machine_def().group_name,
            "current_state_context": current_state_context,
            "history": history_data
        }

    def from_dict(self, id: str,
                  data: Dict[str, Any],
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
            instance_id=current_state_context_date["id"])

        history = [StateContextDescription(
            state_context_builder_full_class_name=state_machine_def.state_context_builder_full_class_name,
            state_def=next(sd for sd in state_machine_def.states_def if sd.name == context_data["state_def_name"]),
            state_machine_definition_repository=state_machine_definition_repository,
            state_machine_repository=state_machine_context_repository,
            instance_id=context_data["id"],
        ) for context_data in data["history"]]

        state_machine = StateMachine(
            id=id,
            state_machine_def_group_name=data["state_machine_def_group_name"],
            state_machine_def_name=data["state_machine_def_name"],
            current_state_context_des=current_state_context,
            history=history,
            state_machine_repository=state_machine_context_repository,
            state_machine_definition_repository=state_machine_definition_repository
        )
        return state_machine

    def form_json(self, id: str,
                  json_text: str,
                  state_machine_definition_repository: StateMachineDefinitionRepository,
                  state_machine_context_repository: StateMachineRepository
                  ) -> "StateMachine":
        data = json.loads(json_text)
        return self.from_dict(id, data, state_machine_definition_repository, state_machine_context_repository)


class ActionRegister:
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


class StateContextRegister:
    _registry = {}

    @classmethod
    def register(cls, full_class_name: str, state_context: StateContext):
        cls._registry[full_class_name] = state_context

    @classmethod
    def get(cls, full_class_name: str) -> StateContext:
        return cls._registry.get(full_class_name)
