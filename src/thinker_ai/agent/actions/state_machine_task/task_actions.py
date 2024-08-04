import json
from typing import Any, List, Tuple

from thinker_ai.agent.actions import Action
from thinker_ai.agent.actions.state_machine_task.composite_task import CompositeTask
from thinker_ai.agent.actions.state_machine_task.task import TaskResult
from thinker_ai.agent.actions.state_machine_task.task_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.llm_schema import Message
from thinker_ai.common.common import replace_curly_braces
from thinker_ai.common.logs import logger
from thinker_ai.status_machine.task_desc import TaskType, TaskDesc
from thinker_ai.utils.code_parser import CodeParser

class ReviewConst:
    TASK_REVIEW_TRIGGER = "task"
    CODE_REVIEW_TRIGGER = "code"
    CONTINUE_WORDS = ["confirm", "continue", "c", "yes", "y"]
    CHANGE_WORDS = ["change"]
    EXIT_WORDS = ["exit"]
    TASK_REVIEW_INSTRUCTION = (
        f"If you want to change, add, delete a task or merge tasks in the plan, say '{CHANGE_WORDS[0]} task task_id or current task, ... (things to change)' "
        f"If you confirm the output from the current task and wish to continue, type: {CONTINUE_WORDS[0]}"
    )
    CODE_REVIEW_INSTRUCTION = (
        f"If you want the codes to be rewritten, say '{CHANGE_WORDS[0]} ... (your change advice)' "
        f"If you want to leave it as is, type: {CONTINUE_WORDS[0]} or {CONTINUE_WORDS[1]}"
    )
    EXIT_INSTRUCTION = f"If you want to terminate the process, type: {EXIT_WORDS[0]}"
class PlanAction(Action):

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def run(self, goal: str, task_name, instruction: str, exec_logger: Memory) -> str:
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


class AskPlanResult(Action):
    async def run(
            self, composite_task: CompositeTask
    ) -> TaskResult:
        PROMPT = """
        # Task Goal
        {user_requirement}

        ## Plan Status
        {plan_status}

        # Instruction
        Summery the plan execution status to answer the goal result
        """
        user_requirement = composite_task.state_def.description
        plan_execution_status = [task.to_dict() for task in composite_task.tasks]
        plan_status = json.dumps(obj=plan_execution_status, indent=4, ensure_ascii=False)
        prompt = PROMPT.format(user_requirement=user_requirement, plan_status=plan_status)
        rsp = await self._aask(prompt)
        return TaskResult(is_success=composite_task.is_finished, result=rsp)


class AskReview(Action):
    async def run(
            self,  task_desc: str,exec_logs: list[Message] = None,trigger: str = ReviewConst.TASK_REVIEW_TRIGGER
    ) -> Tuple[str, bool]:
        if task_desc:
            logger.info("Current overall plan:")
            logger.info(task_desc)

        logger.info("Most recent context:")
        if exec_logs:
            latest_action = exec_logs[-1].cause_by if exec_logs and exec_logs[-1].cause_by else ""
        review_instruction = (
            ReviewConst.TASK_REVIEW_INSTRUCTION
            if trigger == ReviewConst.TASK_REVIEW_TRIGGER
            else ReviewConst.CODE_REVIEW_INSTRUCTION
        )
        prompt = (
            f"This is a <{trigger}> review. Please review output from {latest_action}\n"
            f"{review_instruction}\n"
            f"{ReviewConst.EXIT_INSTRUCTION}\n"
            "Please type your review below:\n"
        )

        rsp = input(prompt)

        if rsp.lower() in ReviewConst.EXIT_WORDS:
            exit()

        # Confirmation can be one of "confirm", "continue", "c", "yes", "y" exactly, or sentences containing "confirm".
        # One could say "confirm this task, but change the next task to ..."
        confirmed = rsp.lower() in ReviewConst.CONTINUE_WORDS or ReviewConst.CONTINUE_WORDS[0] in rsp.lower()

        return rsp, confirmed
