from typing import Any

from thinker_ai.agent.actions import Action

from thinker_ai.agent.memory.memory import Memory
from thinker_ai.app.design.solution.solution_node_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.common.common import replace_curly_braces
from thinker_ai.status_machine.task_desc import TaskType
from thinker_ai.utils.code_parser import CodeParser


class StateMachineDefinitionAction(Action):

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def run(self, group_id: str, state_machine_definition_name: str, description: str,
                  exec_logger: Memory) -> str:
        if not group_id or not state_machine_definition_name or not description:
            raise Exception("group_id and state_machine_definition_name and description must be provided")
        create_or_update = "create"
        state_machine_def = state_machine_definition_repository.get(group_id, state_machine_definition_name)
        if state_machine_def:
            create_or_update = "update"

        PROMPT_TEMPLATE: str = """
        #The Id Of State Machine Definition Group:
        {group_id}
        #The Name Of State Machine Definition:
        {state_machine_definition_name}
        # Create Or Update
         you will {create_or_update} this State Machine Definition
        # Description:
        {description}
        # The Action History:
        {action_history}
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
        IMPORTANT:ONLY the label and description fields will be consistent with the user's local language, the rest of the fields MUST be output in English, please check CAREFUlLY, including group_id, state_machine_definition_name, pre_check_list, post_check_list and so on.
Avoid unnecessary localization.
        """
        prompt = PROMPT_TEMPLATE.format(
            group_id=group_id,
            state_machine_definition_name=state_machine_definition_name,
            create_or_update=create_or_update,
            description=description,
            action_history="\n".join([str(ct) for ct in exec_logger.get()]),
            exist_status_definition=replace_curly_braces(state_machine_definition_repository.group_to_json(group_id)),
            exist_status_scenario=replace_curly_braces(state_machine_scenario_repository.group_to_json(group_id)),
            guidance=TaskType.STATE_MACHINE_PLAN.value.guidance
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp
