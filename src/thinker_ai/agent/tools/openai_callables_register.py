from ctypes import cast
from typing import List, Dict, Callable, Optional, Type

from langchain.agents import load_tools
from langchain_community.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from openai.types import FunctionDefinition
from openai.types.beta import FunctionTool
from pydantic import BaseModel


def callable_to_function_tool(callable: Callable, args_schema: Optional[Type[BaseModel]]) -> FunctionTool:
    # 获取函数名称和描述
    callable_name = get_callable_name(callable)
    function_description = callable.__doc__ or "No description provided."

    # 使用 Pydantic 的 schema() 方法生成 JSON 模式
    parameters = args_schema.schema() if args_schema else None

    # 创建 FunctionDefinition 对象
    function_definition = FunctionDefinition(
        name=callable_name,
        description=function_description,
        parameters=parameters
    )
    return FunctionTool(function=function_definition, type="function")


def get_callable_name(callable: Callable):
    if hasattr(callable, "__self__") and hasattr(callable.__self__, "__class__"):
        # 这是一个实例方法或类方法，获取类名
        class_name = callable.__self__.__class__.__name__
        callable_name = f"{class_name}__{callable.__name__}"
    else:
        # 这是一个普通的函数
        callable_name = callable.__name__
    return callable_name


def tool_to_function_tool(tool: StructuredTool) -> FunctionTool:
    """
    将 StructuredTool 转换为 FunctionTool。

    :param tool: StructuredTool 实例
    :return: FunctionTool 实例
    """
    # 将 StructuredTool 转换为 OpenAI API 可理解的函数定义字典
    openai_function = convert_to_openai_function(tool)

    # 创建 FunctionDefinition 实例
    function_definition = FunctionDefinition(**openai_function)

    # 创建 FunctionTool 实例
    function_tool = FunctionTool(function=function_definition, type="function")

    return function_tool


class CallableRegister:
    """
    为了避免和Assistant中的Tool混淆，并且和FunctionCall一致，命名为CallableRegister
    """

    def __init__(self):
        self.langchain_tools: Dict[str, StructuredTool] = {}

    @property
    def callables_schema(self) -> List[Dict]:
        results: List[Dict] = []
        for langchain_tool in self.langchain_tools.values():
            function_tool = tool_to_function_tool(langchain_tool)
            result = function_tool.function.model_dump()
            results.append(result)
        return results

    @property
    def callable_names(self) -> List[str]:
        results: List[str] = []
        for langchain_tool in self.langchain_tools.values():
            function_tool = tool_to_function_tool(langchain_tool)
            results.append(function_tool.function.name)
        return results

    def register_langchain_tool(self, tool: StructuredTool):
        self.langchain_tools[tool.name] = tool

    def register_langchain_tool_names(self, tool_names: List[str]):
        tools = load_tools(tool_names)
        for tool in tools:
            tool = cast(tool, StructuredTool)
            self.register_langchain_tool(tool)

    def register_callable(self, callable: Callable, args_schema: Optional[Type[BaseModel]]) -> str:
        tool: StructuredTool = StructuredTool.from_function(func=callable, args_schema=args_schema)
        tool.name = get_callable_name(callable)
        self.register_langchain_tool(tool)
        return tool.name

    def get_langchain_tool(self, name: str) -> StructuredTool:
        return self.langchain_tools.get(name)

    def get_function_tool(self, name: str) -> FunctionTool:
        langchain_tool = self.langchain_tools.get(name)
        if langchain_tool:
            return tool_to_function_tool(langchain_tool)

    def clear(self):
        self.langchain_tools.clear()

    def remove_callable(self, name: str):
        if name in self.langchain_tools.keys():
            self.langchain_tools.pop(name)
