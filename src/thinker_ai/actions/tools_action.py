from typing import Callable, Optional, Type, Union, Dict

from langchain.tools import BaseTool
from pydantic import BaseModel

from thinker_ai.actions.action import BaseAction
from thinker_ai.llm.llm_factory import get_llm
from thinker_ai.tools.tools_register import ToolsRegister


class ToolsAction(BaseAction):
    tools_register = ToolsRegister()

    async def act(self,content: str, *args, **kwargs):
        function_call = get_llm().generate_function_call(content, candidate_functions=self.tools_register.tools_schema)
        execute_result = self._call_tool(function_call.name, function_call.arguments)
        rsp_str = await self._generate_answer_from_result(content, execute_result)
        return rsp_str

    def register_tools(self, tool: BaseTool):
        self.tools_register.register(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.tools_register.register_function(func, args_schema)

    def register_langchain_tools(self, tool_names: list[str]):
        self.tools_register.register_langchain_tools(tool_names)

    def _call_tool(self, name: str, arguments: Union[str, Dict]):
        tool = self.tools_register.get_tool(name)
        result = tool.run(arguments)
        return result

    @staticmethod
    async def _generate_answer_from_result(content, execute_result) -> str:
        result_prompt_template = "用户提出的问题是“{content}”,现在已经有了数据:“{execute_result}”，请用这个数据提供自然语言的回复"
        prompt = result_prompt_template.format(content=content, execute_result=execute_result)
        rsp_str = await get_llm().a_generate(prompt)
        return rsp_str
