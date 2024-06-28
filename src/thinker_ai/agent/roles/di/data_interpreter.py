from __future__ import annotations

import json
from typing import Literal

from pydantic import model_validator
from thinker_ai.agent.actions.di.task_tree import TaskTree
from thinker_ai.agent.actions.di.tool_recommend import ToolRecommender
from thinker_ai.agent.provider.schema import Message
from thinker_ai.agent.actions.di.write_analysis_code import WriteAnalysisCode
from thinker_ai.agent.roles import Role
from thinker_ai.agent.roles.role import RoleReactMode
from thinker_ai.utils.code_parser import CodeParser

REACT_THINK_PROMPT = """
# User Requirement
{user_requirement}
# Context
{context}

Output a json following the format:
```json
{{
    "thoughts": str = "Thoughts on current situation, reflect on how you should proceed to fulfill the user requirement",
    "state": bool = "Decide whether you need to take more actions to complete the user requirement. Return true if you think so. Return false if you think the requirement has been completely fulfilled."
}}
```
"""


class DataInterpreter(Role):
    name: str = "David"
    profile: str = "DataInterpreter"
    auto_run: bool = True
    use_reflection: bool = False
    tools: list[str] = []  # Use special symbol ["<all>"] to indicate use of all registered tools
    tool_recommender: ToolRecommender = None
    react_mode: Literal["plan_and_act", "react"] = "plan_and_act"
    max_react_loop: int = 10  # used for react mode


    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "Interpreter":
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop)
        self.set_actions([WriteAnalysisCode])
        self._set_state(0)
        return self

    @property
    def working_memory(self):
        return self.rc.working_memory

    async def _think(self) -> bool:
        """Useful in 'react' mode. Use LLM to decide whether and what to do next."""
        user_requirement = self.get_memories()[0].content
        context = self.working_memory.get()

        if not context:
            # just started the run, we need action certainly
            self.working_memory.add(self.get_memories()[0])  # add user requirement to working memory
            self._set_state(0)
            return True

        prompt = REACT_THINK_PROMPT.format(user_requirement=user_requirement, context=context)
        rsp = await self.llm.aask(prompt)
        rsp_dict = json.loads(CodeParser.parse_code(block=None, text=rsp))
        self.working_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))
        need_action = rsp_dict["state"]
        self._set_state(0) if need_action else self._set_state(-1)

        return need_action

    async def react(self) -> Message:
        """Entry to one of three strategies by which Role reacts to the observed Message"""
        if self.rc.react_mode == RoleReactMode.REACT or self.rc.react_mode == RoleReactMode.BY_ORDER:
            rsp = await self._react()
        elif self.rc.react_mode == RoleReactMode.PLAN_AND_ACT:
            rsp = await self._plan_and_act()
        else:
            raise ValueError(f"Unsupported react mode: {self.rc.react_mode}")
        self._set_state(state=-1)  # current reaction is complete, reset state to -1 and todo back to None
        return rsp

    async def _plan_and_act(self) -> Message:
        """first plan, then execute an action sequence, i.e. _think (of a plan) -> _act -> _act -> ... Use llm to come up with the plan dynamically."""
        # create initial plan and update it until confirmation
        instruction = self.rc.memory.get()[-1].content  # retreive latest user requirement
        task_tree = TaskTree(
            id="tree_root",
            instruction=instruction,
            use_reflection=self.use_reflection,
            tools=self.tools,
            auto_run=self.auto_run,
            tool_recommender=self.tool_recommender
        )
        await task_tree.execute_plan()
        result_msg=Message(role="assistant", content=task_tree.task_result.result)
        self.rc.memory.add(result_msg)  # add to persistent memory
        return result_msg
