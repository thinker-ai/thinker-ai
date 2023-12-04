from typing import Optional, List, Dict

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel

from thinker_ai.llm.function_call import FunctionCall


class ClaudeConfig(BaseModel):
    api_key: str
    max_tokens: int = 1000
    model: str = "claude-2"


class Claude2:

    def __init__(self, config: ClaudeConfig):
        self.max_tokens = config.max_tokens
        self.anthropic = Anthropic(api_key=config.api_key)

    def generate(self, model: str, prompt, system_prompt: Optional[list[str]] = None) -> str:
        return self.anthropic.completions.create(
            model=model,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=self.max_tokens,
        ).completion

    async def a_generate(self, model: str, prompt, system_prompt: Optional[list[str]] = None) -> str:
        return self.anthropic.completions.create(
            model=model,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=self.max_tokens,
        ).completion

    async def a_generate_stream(self, model: str, prompt, system_prompt: Optional[list[str]] = None) -> str:
        raise NotImplementedError()

    def generate_function_call(self, model: str, prompt, candidate_functions: List[Dict],
                               system_prompt: Optional[list[str]] = None) -> FunctionCall:
        raise NotImplementedError()

    async def a_generate_function_call(self, model: str, prompt, candidate_functions: List[Dict],
                                       system_prompt: Optional[list[str]] = None) -> FunctionCall:
        raise NotImplementedError()

    def generate_batch(self, model: str, msgs) -> str:
        raise NotImplementedError()