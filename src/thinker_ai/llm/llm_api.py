
from abc import abstractmethod, ABC
from typing import List, Dict, Optional
from pydantic import BaseModel


class FunctionCall(BaseModel):
    """The base model for function call"""
    name: str
    arguments: Dict


class LLM_API(ABC):
    @abstractmethod
    def generate(self, user_msg:str,system_msg: Optional[str] = None) -> str:
        pass

    @abstractmethod
    async def a_generate(self, user_msg:str,system_msg: Optional[str] = None,stream=False) -> str:
        pass

    @abstractmethod
    async def a_generate_batch(self, user_msgs: list[str], sys_msg: Optional[str]= None) -> list[str]:
        pass

    @abstractmethod
    def generate_function_call(self, user_msg:str, candidate_functions: List[Dict], sys_msg: Optional[str] = None)-> FunctionCall:
        pass

    @abstractmethod
    async def a_generate_function_call(self, user_msg:str, candidate_functions: List[Dict],sys_msg: Optional[str] = None) -> FunctionCall:
        pass
