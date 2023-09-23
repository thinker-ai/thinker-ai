import os
from typing import Optional, List, Dict

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel

from thinker_ai.llm.llm_api import LLM_API, FunctionCall
from thinker_ai.utils import Singleton


class ClaudeConfig(BaseModel):
    api_key: str
    max_tokens: int = 1000
    model: str = "claude-2"


class Claude2(LLM_API, metaclass=Singleton):

    def __init__(self,config: ClaudeConfig):
        self.max_tokens = config.max_tokens
        self.model = config.model
        self.anthropic = Anthropic(api_key=config.api_key)

    def generate(self, prompt,system_prompt: Optional[list[str]] = None) -> str:
        return self.anthropic.completions.create(
            model=self.model,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=self.max_tokens,
        ).completion

    async def a_generate(self, prompt,system_prompt: Optional[list[str]] = None) -> str:
        return self.anthropic.completions.create(
            model=self.model,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=self.max_tokens,
        ).completion

    async def a_generate_stream(self, prompt,system_prompt: Optional[list[str]] = None) -> str:
        raise NotImplementedError()

    def generate_function_call(self, prompt, candidate_functions: List[Dict],
                               system_prompt: Optional[list[str]] = None) -> FunctionCall:
        raise NotImplementedError()

    async def a_generate_function_call(self, prompt, candidate_functions: List[Dict],
                                 system_prompt: Optional[list[str]] = None) -> FunctionCall:
        raise NotImplementedError()

    def generate_batch(self, msgs)-> str:
        raise NotImplementedError()