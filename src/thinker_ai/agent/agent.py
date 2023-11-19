from abc import abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel

from thinker_ai.actions.action import Criteria, BaseAction
from thinker_ai.llm.llm_factory import get_llm


class DataModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __iter__(self):
        return iter(self.__dict__.values())


class Agent:

    def __init__(self, name: str, actions: Dict[str, BaseAction] = None):
        self.name = name
        self.actions = actions
        if self.actions is None:
            self.actions = {}

    async def ask(self, content: str) -> DataModel:
        rsp_str = await get_llm().a_generate(content)
        return DataModel(rsp_str=rsp_str)
