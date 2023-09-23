
from abc import abstractmethod, ABC
from typing import List, Dict, Optional

from pydantic import BaseModel


class FunctionCall(BaseModel):
    """The base model for function call"""
    name: str
    arguments: Dict


class LLM_API(ABC):
    @abstractmethod
    def generate(self, prompt,system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def generate_function_call(self, prompt, candidate_functions: List[Dict], system_prompt: Optional[str] = None)-> FunctionCall:
        pass

    @abstractmethod
    async def a_generate(self, prompt, system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    async def a_generate_function_call(self, prompt, candidate_functions: List[Dict],system_prompt: Optional[str] = None) -> FunctionCall:
        pass

    @abstractmethod
    async def a_generate_stream(self, prompt,system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def generate_batch(self, msgs)-> str:
        pass



