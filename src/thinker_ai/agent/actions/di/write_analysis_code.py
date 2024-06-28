from __future__ import annotations

import json

from thinker_ai.agent.actions import Action
from thinker_ai.agent.actions.di.task_desc import TaskDesc, PlanStatus
from thinker_ai.agent.provider.schema import Message
from thinker_ai.agent.prompts.di.write_analysis_code import (
    DEBUG_REFLECTION_EXAMPLE,
    INTERPRETER_SYSTEM_MSG,
    REFLECTION_PROMPT,
    REFLECTION_SYSTEM_MSG,
    STRUCTUAL_PROMPT,
)
from thinker_ai.utils.code_parser import CodeParser


class WriteAnalysisCode(Action):
    async def _debug_with_reflection(self, exec_logs: list[Message]):
        reflection_prompt = REFLECTION_PROMPT.format(
            debug_example=DEBUG_REFLECTION_EXAMPLE,
            previous_impl="\n\n".join(f"{log.role}:\n{log.content}" for log in exec_logs)
        )

        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        json_str = CodeParser.parse_code(block=None, text=rsp)
        reflection = json.loads(json_str)
        return reflection["improved_impl"]

    async def run(
            self,
            parent_task_desc: TaskDesc,
            plan_status: PlanStatus,
            task_desc: TaskDesc,
            exec_logs: list[Message],
            tool_info: str,
            use_reflection: bool = False,
            **kwargs: object,
    ) -> str:

        context = STRUCTUAL_PROMPT.format(
            parent_task_instruction=parent_task_desc.instruction,
            parent_plan=json.dumps(parent_task_desc.plan, indent=4,ensure_ascii=False),
            plan_status=plan_status.to_string(),
            task_desc=task_desc.to_string(),
            tool_info=tool_info,
        )
        # LLM call
        if use_reflection:
            exec_logs = exec_logs or []
            code = await self._debug_with_reflection(exec_logs=exec_logs)
        else:
            rsp = await self.llm.aask(msg=context, system_msgs=[INTERPRETER_SYSTEM_MSG], **kwargs)
            code = CodeParser.parse_code(block=None, text=rsp)

        return code
