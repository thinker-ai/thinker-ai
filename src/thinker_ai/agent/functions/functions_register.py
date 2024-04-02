from typing import List, Dict, Callable, Optional, Type, Any

from langchain.agents import load_tools
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
        results: List[Dict] = []
        for tool_description in self.functions_dict.values():
            result = format_tool_to_openai_function(tool_description)
            results.append(result)
        return results

    @property
    def functions(self) -> List[BaseTool]:
        return list(self.functions_dict.values())

    def register_langchain_tool(self, function: BaseTool):
        self.functions_dict[function.name] = function

    def register_langchain_tool_names(self, tool_names: List[str]):
        tools = load_tools(tool_names)
        for tool in tools:
            self.register_langchain_tool(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        base_tool: BaseTool = StructuredTool.from_function(func=func, args_schema=args_schema)
        self.register_langchain_tool(base_tool)

    def get_function(self, name: str) -> BaseTool:
        return self.functions_dict.get(name)

    def clear(self):
        self.functions_dict.clear()