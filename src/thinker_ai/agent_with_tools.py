import re
from typing import Callable, Union, Optional, Type, Any, Dict, List
from langchain import OpenAI
from langchain.agents import load_tools
from langchain.tools import format_tool_to_openai_function
from langchain.tools.base import BaseTool, StructuredTool
from pydantic import BaseModel

from thinker_ai.context import Context
from thinker_ai.llm.llm_factory import get_llm

class ToolsRegister:

    def __init__(self):
        self.tool_descriptions: Dict[str, BaseTool] = {}

    @property
    def tools_schema(self) -> List[Dict]:
        return [format_tool_to_openai_function(tool_description)
                for tool_description in self.tool_descriptions.values()]

    @property
    def tools(self) -> List[BaseTool]:
        return list(self.tool_descriptions.values())

    def register(self, tool: BaseTool):
        self.tool_descriptions[tool.name] = tool

    def get_tool(self, name: str):
        return self.tool_descriptions[name]

    def clear(self):
        self.tool_descriptions.clear()


class LLM_Response(BaseModel):
    content: str


class AgentWithTools:
    tools_register = ToolsRegister()

    def __init__(self,context:Context):
        self.context = context

    def get_tools_schema(self) -> List[Dict]:
        return self.tools_register.tools_schema

    def register_tool(self, tool: BaseTool):
        self.tools_register.register(tool)

    def register_langchain_tools(self, tools_name: list[str]):
        tools = load_tools(tools_name, OpenAI(temperature=0))
        for tool in tools:
            self.register_tool(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        base_tool: BaseTool = StructuredTool.from_function(func=func, args_schema=args_schema)
        self.register_tool(base_tool)

    def call_tool(self, name: str, arguments: Union[str, Dict]):
        tool = self.tools_register.get_tool(name)
        result = tool.run(arguments)
        return result

    def chat_with_function_call(self, content: str) -> Any:
        function_call = get_llm().generate_function_call(content, candidate_functions=self.tools_register.tools_schema)
        execute_result = self.call_tool(function_call.name, function_call.arguments)
        response = self._generate_answer_from_result(content, execute_result)
        return response

    def _generate_answer_from_result(self, content, execute_result):
        result_prompt_template = "用户提出的问题是“{content}”,现在已经有了数据:“{execute_result}”，请用这个数据提供自然语言的回复"
        prompt = result_prompt_template.format(content=content, execute_result=execute_result)
        response = get_llm().generate(prompt)
        return response

    def _execute_generated_code(self, generated_code: str) -> Any:
        try:
            locals_dict = {}
            exec(generated_code, globals(), locals_dict)
            if 'output_data' in locals_dict:
                return locals_dict['output_data']
        except Exception as e:
            # Print the exception message for debugging purposes
            print("Exception occurred:", e)
            raise e

    def _extract_generated_code(self, llm_response: LLM_Response) -> Optional[str]:
        code_pattern = re.compile(r"##code:\n'''([\s\S]*?)'''")
        code_texts = code_pattern.findall(llm_response.content)
        if not code_texts:
            code_pattern = re.compile(r"##code:\n'''python([\s\S]*?)'''")
            code_texts = code_pattern.findall(llm_response.content)
        if not code_texts:
            return None
        return code_texts[0]

    def _extract_generated_think(self, llm_response: LLM_Response):
        think_pattern = re.compile(r"##think:\n'''([\s\S]*?)'''")
        think_texts = think_pattern.findall(llm_response.content)
        if not think_texts:
            raise "think_text not found"
        return think_texts[0]
