from typing import List, Dict, Callable, Optional, Type

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.tools import format_tool_to_openai_function, BaseTool, StructuredTool
from pydantic import BaseModel


class ToolsRegister:

    def __init__(self):
        self.tools_dict: Dict[str, BaseTool] = {}

    @property
    def tools_schema(self) -> List[Dict]:
        return [format_tool_to_openai_function(tool_description)
                for tool_description in self.tools_dict.values()]

    @property
    def tools(self) -> List[BaseTool]:
        return list(self.tools_dict.values())

    def register(self, tool: BaseTool):
        self.tools_dict[tool.name] = tool

    def register_langchain_tools(self, tools_name: list[str]):
        tools = load_tools(tools_name, OpenAI(temperature=0))
        for tool in tools:
            self.register(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        base_tool: BaseTool = StructuredTool.from_function(func=func, args_schema=args_schema)
        self.register(base_tool)

    def get_tool(self, name: str) -> BaseTool:
        return self.tools_dict[name]

    def clear(self):
        self.tools_dict.clear()




