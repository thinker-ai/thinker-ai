from typing import Any

from thinker_ai.agent.actions import Action

from thinker_ai.agent.memory.memory import Memory
from thinker_ai.app.design.solution.solution_node_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.common.common import replace_curly_braces
from thinker_ai.status_machine.task_desc import TaskType
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

    async def run(self, goal_name: str, task_name, instruction: str, exec_logger: Memory) -> str:
        if not task_name or not instruction:
            raise Exception("task_name and instruction must be provided")
        create_or_update = "create"
        if not goal_name:
            goal_name = task_name
        state_machine_def = state_machine_definition_repository.get(goal_name, task_name)
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
        IMPORTANT:ONLY the label and description fields will be consistent with the user's local language, the rest of the fields MUST be output in English, please check CAREFUlLY, including group_name, state_machine_definition_name, pre_check_list, post_check_list and so on.
Avoid unnecessary localization.
        """

        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = PROMPT_TEMPLATE.format(
            plan_name=task_name,
            goal=goal_name,
            create_or_update=create_or_update,
            instruction=instruction,
            scenario="\n".join([str(ct) for ct in exec_logger.get()]),
            task_type_desc=task_type_desc,
            exist_status_definition=replace_curly_braces(state_machine_definition_repository.group_to_json(goal_name)),
            exist_status_scenario=replace_curly_braces(state_machine_scenario_repository.group_to_json(goal_name)),
            guidance=TaskType.STATE_MACHINE_PLAN.value.guidance
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp
