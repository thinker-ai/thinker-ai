from typing import Callable, Optional, Type, Union, Dict

from langchain.tools import BaseTool
from pydantic import BaseModel

from thinker_ai.agent.agent import Agent
from thinker_ai.agent.llm import gpt
from thinker_ai.agent.functions.functions_register import FunctionsRegister


class FunctionCallAgent(Agent):

    tools_register = FunctionsRegister()

    async def act(self,model:str,content: str, *args, **kwargs):
        function_call = gpt.generate_function_call(model, content, candidate_functions=self.tools_register.functions_schema)
        execute_result = self._call_function(function_call.name, function_call.arguments)
        rsp_str = await self._generate_answer_from_result(model,content, execute_result)
        return rsp_str

    def register_tools(self, tool: BaseTool):
        self.tools_register.register(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.tools_register.register_function(func, args_schema)

    def register_langchain_functions(self, tool_names: list[str]):
        self.tools_register.register_langchain_tools(tool_names)

    def _call_function(self, name: str, arguments: Union[str, Dict]):
        tool = self.tools_register.get_function(name)
        result = tool.run(arguments)
        return result

    @staticmethod
    async def _generate_answer_from_result(model:str,content, execute_result) -> str:
        result_prompt_template = "用户提出的问题是“{content}”,现在已经有了数据:“{execute_result}”，请用这个数据提供自然语言的回复"
        prompt = result_prompt_template.format(content=content, execute_result=execute_result)
        rsp_str = await gpt.a_generate(model,prompt)
        return rsp_str
