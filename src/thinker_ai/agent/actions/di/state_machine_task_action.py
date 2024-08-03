import json
import uuid
from typing import Optional, Any, List, Tuple

from thinker_ai.agent.actions import Action
from thinker_ai.agent.actions.di.state_machine_task import StateMachineTaskScenario
from thinker_ai.agent.actions.di.state_machine_task_repository import state_machine_definition_repository, state_machine_scenario_repository
from thinker_ai.agent.actions.di.state_task import StateTask, StateTaskResult
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.common.common import replace_curly_braces
from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_definition import StateDefinition, BaseStateDefinition, StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import StateScenarioBuilder, StateScenario, BaseStateScenario, \
    StateMachineRepository, StateMachineContextBuilder
from thinker_ai.status_machine.task_desc import TaskType
from thinker_ai.utils.code_parser import CodeParser


class TaskBuilder(StateScenarioBuilder):
    @staticmethod
    def build_state_scenario(state_def: StateDefinition,
                             state_scenario_class_name: str,
                             scenario_id: Optional[str] = str(uuid.uuid4())) -> StateScenario:
        return StateTask(scenario_id=scenario_id, state_def=state_def)

    @staticmethod
    def build_terminal_state_scenario(state_def: BaseStateDefinition,
                                      state_scenario_class_name: str,
                                      scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        return BaseStateScenario(scenario_id=scenario_id, state_def=state_def)

    @classmethod
    def build_composite_state_scenario(cls, state_def: StateDefinition,
                                       state_scenario_class_name: str,
                                       state_machine_definition_repository: StateMachineDefinitionRepository,
                                       state_machine_scenario_repository: StateMachineRepository,
                                       scenario_group_id: str,
                                       scenario_id: Optional[str] = str(uuid.uuid4())) -> BaseStateScenario:
        if state_def.is_composite:
            return StateMachineTaskScenario(scenario_id=scenario_id,
                                            scenario_group_id=scenario_group_id,
                                            state_def=state_def,
                                            state_scenario_builder_class=cls,
                                            state_machine_scenario_repository=state_machine_scenario_repository,
                                            state_machine_definition_repository=state_machine_definition_repository
                                            )


class StateMachineTaskPlanAction(Action):

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def run(self, goal: str, task_name, instruction: str,exec_logger: Memory) -> str:
        if not task_name or not instruction:
            raise Exception("task_name and instruction must be provided")
        create_or_update = "create"
        if not goal:
            goal = task_name
        state_machine_def = state_machine_definition_repository.get(goal, task_name)
        if state_machine_def:
            create_or_update = "update"

        PROMPT_TEMPLATE: str = """
        #The Name Of State Machine Group:
        {goal}
        #The Name Of State Machine:
        {plan_name}
        # Create Or Update
         you will {create_or_update} this State Machine
        # Instruction:
        {instruction}
        # Context:
        {scenario}
        # Available Task Types:
        {task_type_desc}
        # Guidance:
        the exist state machine definitions are here,they are related to the current state machine and are the scenario of the current state machine:
        ```json
         {exist_status_definition}
        ```
        the exist state machine scenarios are here,they are related to the current state machine and are the scenario of the current state machine:
        ```json
         {exist_status_scenario}
        ```
        {guidance}
        """

        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = PROMPT_TEMPLATE.format(
            plan_name=task_name,
            goal=goal,
            create_or_update=create_or_update,
            instruction=instruction,
            scenario="\n".join([str(ct) for ct in exec_logger.get()]),
            task_type_desc=task_type_desc,
            exist_status_definition=replace_curly_braces(state_machine_definition_repository.group_to_json(goal)),
            exist_status_scenario=replace_curly_braces(state_machine_scenario_repository.group_to_json(goal)),
            guidance=TaskType.STATE_MACHINE_PLAN.value.guidance
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp

    def pre_check_plan_from_rsp(self, rsp: str, goal, task_name) -> Tuple[bool, str]:
        try:
            state_machine = (StateMachineContextBuilder
                             .new_from_group_def_json(state_machine_def_group_name=goal,
                                                      state_machine_def_name=task_name,
                                                      def_json=rsp,
                                                      state_machine_definition_repository=state_machine_definition_repository,
                                                      state_machine_scenario_repository=state_machine_scenario_repository)
                             )

            if state_machine:
                results: List[Tuple[List[Command], bool]] = state_machine.self_validate()
                success = True
                fail_paths = []
                for command_list, result in results:
                    if not result:
                        fail_path = []
                        for command in command_list:
                            fail_path.append((command.name, command.target))
                            fail_paths.append(fail_path)
                        success = result
                if success:
                    return True, "状态机验证成功。"
                else:
                    return False, str(fail_paths)
        except Exception as e:
            return False, str(e)


class AskPlanResult(Action):
    async def run(
            self, state_machine_task_scenario: StateMachineTaskScenario
    ) -> StateTaskResult:
        PROMPT = """
        # Task Goal
        {user_requirement}

        ## Plan Status
        {plan_status}

        # Instruction
        Summery the plan execution status to answer the goal result
        """
        user_requirement = state_machine_task_scenario.instruction
        plan_execution_status = [task.to_dict() for task in state_machine_task_scenario.tasks]
        plan_status = json.dumps(obj=plan_execution_status, indent=4, ensure_ascii=False)
        prompt = PROMPT.format(user_requirement=user_requirement, plan_status=plan_status)
        rsp = await self._aask(prompt)
        return StateTaskResult(is_success=state_machine_task_scenario.is_finished, result=rsp)
