"""RAG LLM."""
import asyncio
from typing import Any

from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

from thinker_ai.agent.provider.llm import LLM
from thinker_ai.agent.provider.token_counter import TOKEN_MAX
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.configs.config import config
from thinker_ai.utils.async_helper import NestAsyncio


class RAGLLM(CustomLLM):
    """LlamaIndex's LLM is different from ThinkerAI's LLM.

    Inherit CustomLLM from llamaindex, making ThinkerAI's LLM can be used by LlamaIndex.
    """

    model_infer: BaseLLM = Field(..., description="The ThinkerAI's LLM.")
    context_window: int = TOKEN_MAX.get(config.client.model, DEFAULT_CONTEXT_WINDOW)
    num_output: int = config.client.max_token
    model_name: str = config.client.model

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output, model_name=self.model_name or "unknown"
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        NestAsyncio.apply_once()
        return asyncio.get_event_loop().run_until_complete(self.acomplete(prompt, **kwargs))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text = await self.model_infer.aask(msg=prompt, stream=False)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        ...


def get_rag_llm(model_infer: BaseLLM = None) -> RAGLLM:
    """Get llm that can be used by LlamaIndex."""
    return RAGLLM(model_infer=model_infer or LLM())
