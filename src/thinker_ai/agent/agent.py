from abc import abstractmethod
from typing import Dict

from thinker_ai.actions.action import Criteria, BaseAction
from thinker_ai.llm.llm_factory import get_llm


class Agent:

    def __init__(self, name: str, actions: Dict[str, BaseAction]):
        self.name = name
        self.actions = actions

    async def ask(self, content: str) -> str:
        rsp_str = await get_llm().a_generate(content)
        return rsp_str

