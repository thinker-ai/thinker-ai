from typing import List, Dict, Callable, Optional, Type

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.tools import format_tool_to_openai_function, BaseTool, StructuredTool
from pydantic import BaseModel


class FunctionsRegister:
    """
       为了避免和Assistant中的Tool混淆，并且和FunctionCall一致，命名为FunctionsRegister
    """
    def __init__(self):
        self.functions_dict: Dict[str, BaseTool] = {}

    @property
    def functions_schema(self) -> List[Dict]:
        return [format_tool_to_openai_function(tool_description)
                for tool_description in self.functions_dict.values()]

    @property
    def functions(self) -> List[BaseTool]:
        return list(self.functions_dict.values())

    def register(self, function: BaseTool):
        self.functions_dict[function.name] = function

    def register_langchain_tools(self, tools_name: list[str]):
        tools = load_tools(tools_name, OpenAI(temperature=0))
        for tool in tools:
            self.register(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        base_tool: BaseTool = StructuredTool.from_function(func=func, args_schema=args_schema)
        self.register(base_tool)

    def get_function(self, name: str) -> BaseTool:
        return self.functions_dict[name]

    def clear(self):
        self.functions_dict.clear()




