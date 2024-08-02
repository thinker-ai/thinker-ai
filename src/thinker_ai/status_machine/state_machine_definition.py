import json
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, Set, Tuple

from thinker_ai.status_machine.base import Command, Event, ActionDescription
from thinker_ai.status_machine.task_desc import TaskTypeDef, TaskType


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
    def __init__(self, name: str, group_def_name: str, description: str, task_type: TaskTypeDef,
                 state_context_class_name: str,
                 actions_des: Set[ActionDescription], result_events: Set[str], is_start: bool = False,
                 is_composite: bool = False):
        super().__init__(name, description, state_context_class_name)
        self.group_def_name = group_def_name
        self.task_type = task_type
        self.is_start = is_start
        self.events: Set[str] = result_events
        self.actions_des: Set[ActionDescription] = actions_des
        self.is_composite = is_composite


class Transition:
    def __init__(self, event: str, source: BaseStateDefinition, target: BaseStateDefinition):
        self.event = event
        self.source = source
        self.target = target


class StateMachineDefinition:
    def __init__(self,
                 group_def_name: str,
                 name: str,
                 states_def: Set[BaseStateDefinition],
                 transitions: Set[Transition],
                 state_context_builder_full_class_name: Optional[str],
                 is_root: bool = False,
                 inner_end_state_to_outer_event: Optional[Dict[str, str]] = None):
        self.name = name
        self.state_context_builder_full_class_name = state_context_builder_full_class_name
        self.group_def_name = group_def_name
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
            state_machine_def_group = self.group_def_name
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
                    and state_def_name not in state_machine_def_repo.get_state_machine_names(self.group_def_name)
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
    def _from_event_to_mock_action(state: StateDefinition, event: str) -> Optional[ActionDescription]:
        for action in state.actions_des:
            if event.startswith(f"{action.on_command}_result"):  # Assuming event name is used as command in mock action
                return action
        return None

    def _get_state_validate_paths(self) -> List[List[Tuple[str, BaseStateDefinition]]]:
        class Branch:
            def __init__(self, from_state: str):
                self.from_state = from_state
                self.to = {}  # Dictionary to store target states and their visited status

            def add_to_state(self, to_state: str):
                self.to[to_state] = False  # Initially, set the visited status to False

            def mark_visited(self, to_state: str, visited: bool):
                if to_state in self.to:
                    self.to[to_state] = visited

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

        def is_unvisited(branch) -> bool:
            for to_state, visited in branch.to.items():
                if not visited:
                    return True
            return False

        def has_unvisited_branch() -> bool:
            for branch in branches.values():
                if is_unvisited(branch):
                    return True
            return False

        def un_mark_path(path: List[Tuple[str, BaseStateDefinition]]) -> bool:
            un_mark_next = False
            to = None
            for _, state in path[::-1]:
                branch: Branch = branches.get(state.name)
                if branch:
                    if un_mark_next:
                        branch.mark_visited(to, False)
                    elif is_unvisited(branch):
                        un_mark_next = True
                to = state.name
            return True

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

            def visit(state: BaseStateDefinition):
                state_def_name = f"{self.name}.{state.name}"
                current_path.append((state_def_name, state))
                if state.is_terminal():
                    un_mark_path(current_path)
                    return
                for transition in self.transitions:
                    if transition.source.name == state.name:
                        next_state = transition.target
                        branch: Branch = branches.get(transition.source.name)
                        if branch:  # 存在分支
                            visited = branch.to.get(next_state.name)
                            if visited:  # 且访问过
                                continue
                            else:
                                branch.mark_visited(next_state.name, True)
                        return visit(next_state)  # 表示退出for循环

            start_state = self.get_start_state_def()
            if start_state:
                visit(start_state)
                execution_plan.append(current_path)

            # Check if there are any unvisited branches left
            if not has_unvisited_branch():
                break

        return merger_paths(execution_plan)


class StateMachineDefinitionBuilder:

    @classmethod
    def root_from_group_def_json(cls, group_def_name: str, groups_def_json: str) -> StateMachineDefinition:
        groups_def_dict = json.loads(groups_def_json)
        group_def_data = groups_def_dict.get(group_def_name)
        if group_def_data:
            return cls.root_from_group_def_dict(group_def_name, group_def_data)

    @classmethod
    def root_from_group_def_dict(cls, group_def_name: str, group_def_dict: dict) -> StateMachineDefinition:
        for name, state_machine_def in iter(group_def_dict.items()):
            if state_machine_def.get("is_root"):
                return cls.from_dict(group_def_name, name, state_machine_def, set(group_def_dict.keys()))

    @classmethod
    def from_group_def_json(cls, group_def_name: str, state_machine_def_name: str,
                            groups_def_json: str) -> StateMachineDefinition:
        groups_def_dict_data = json.loads(groups_def_json)
        group_def_data = groups_def_dict_data.get(group_def_name)
        if group_def_data:
            state_machine_def_data = group_def_data.get(state_machine_def_name)
            if state_machine_def_data:
                return cls.from_dict(group_def_name, state_machine_def_name, state_machine_def_data,
                                     set(group_def_data.keys()))

    @classmethod
    def from_json(cls, group_def_name: str, state_machine_def_name: str, state_machine_def_json: str,
                  exist_state_machine_names: Set):
        groups_def_dict_data = json.loads(state_machine_def_json)
        return cls.from_dict(group_def_name, state_machine_def_name, groups_def_dict_data, exist_state_machine_names)

    @classmethod
    def from_dict(cls, group_def_name: str, state_machine_def_name: str, state_machine_def_dict: Dict[str, Any],
                  exist_state_machine_names: Set) -> StateMachineDefinition:
        states = {cls._state_def_from_dict(data=sd,
                                           group_def_name=group_def_name,
                                           state_machine_def_name=state_machine_def_name,
                                           exist_state_machine_names=exist_state_machine_names)
                  for sd in state_machine_def_dict["states_def"]}
        transitions = {cls._transition_from_dict(t, states) for t in state_machine_def_dict["transitions"]}
        is_root = True if state_machine_def_dict.get("is_root") else False
        return StateMachineDefinition(
            name=state_machine_def_name,
            group_def_name=group_def_name,
            is_root=is_root,
            state_context_builder_full_class_name=state_machine_def_dict.get("state_context_builder_full_class_name"),
            states_def=states,
            transitions=transitions,
            inner_end_state_to_outer_event=state_machine_def_dict.get(
                "inner_end_state_to_outer_event") if not is_root else None
        )

    @classmethod
    def to_dict(cls, state_machine_def: StateMachineDefinition) -> Dict[str, Any]:
        result = {
            "is_root": state_machine_def.is_root,
            "state_context_builder_full_class_name": state_machine_def.state_context_builder_full_class_name,
            "states_def": [cls._state_def_to_dict(sd) for sd in
                           state_machine_def.states_def],
            "transitions": [cls._transition_to_dict(t) for t in
                            state_machine_def.transitions],
        }
        if not state_machine_def.is_root and state_machine_def.inner_end_state_to_outer_event:
            result["inner_end_state_to_outer_event"] = state_machine_def.inner_end_state_to_outer_event
        return {state_machine_def.name: result}

    @classmethod
    def _state_def_to_dict(cls, state_def: BaseStateDefinition) -> Dict[str, Any]:
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

    @classmethod
    def _state_def_from_dict(cls, data: Dict[str, Any],
                             group_def_name: str,
                             state_machine_def_name: str,
                             exist_state_machine_names: Set) -> BaseStateDefinition:
        is_composite = cls.is_composite_state(parent_state_machine_def_name=state_machine_def_name,
                                              state_def_name=data["name"],
                                              exist_state_machine_names=exist_state_machine_names)
        if data.get("actions"):
            return StateDefinition(
                name=data["name"],
                group_def_name=group_def_name,
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

    @classmethod
    def is_composite_state(cls, parent_state_machine_def_name: str, state_def_name: str, exist_state_machine_names):
        state_machine_name = state_def_name
        if parent_state_machine_def_name:
            state_machine_name = f"{parent_state_machine_def_name}.{state_def_name}"
        return state_machine_name in exist_state_machine_names

    @classmethod
    def _transition_to_dict(cls, transition: Transition) -> Dict[str, str]:
        return {
            "event": transition.event,
            "source": transition.source.name,
            "target": transition.target.name
        }

    @classmethod
    def _transition_from_dict(cls, data: Dict[str, str], states: Set[BaseStateDefinition]) -> Transition:
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
