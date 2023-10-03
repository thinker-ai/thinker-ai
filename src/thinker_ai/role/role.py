from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Dict, List, Union, Any

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel

from thinker_ai.llm.llm_factory import get_llm
from thinker_ai.task.task import Task
from thinker_ai.tools.tools_agent import ToolsAgent
from thinker_ai.tools.tools_register import ToolsRegister


class RoleConfig(BaseModel):
    name: str
    module_name: str
    class_name: str
    langchain_tool_names: list[str] = []
    tools: list[BaseTool] = []
    funcs: Dict[Callable, Optional[Type[BaseModel]]] = {}


class ChatRole:
    tools_agent = ToolsAgent()

    def __init__(self, role_config: RoleConfig):
        not self = role_config.name
        tools = load_tools(role_config.langchain_tool_names, OpenAI(temperature=0))
        for tool in tools:
            self.tools_agent.register(tool)
        for tool in role_config.tools:
            self.tools_agent.register(tool)
        for func in role_config.funcs:
            self.tools_agent.register_function(func)

    @staticmethod
    async def ask(content: str) -> str:
        rsp_str = await get_llm().a_generate(content)
        return rsp_str

    async def ask_with_tools(self, content: str) -> str:
        rsp_str = await self.tools_agent.a_ask_with_function_call(content)
        return rsp_str


class TaskRole(ChatRole):

    def __init__(self, role_config: RoleConfig):
        super().__init__(role_config)

    @abstractmethod
    async def do_task(self, task: Task, *args, **kwargs):
        raise NotImplementedError
