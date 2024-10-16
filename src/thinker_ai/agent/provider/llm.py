from typing import Optional
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.configs.llm_config import LLMConfig
from thinker_ai.context import Context


def LLM(llm_config: Optional[LLMConfig] = None, context: Context = None) -> BaseLLM:
    """get the default llm provider if name is None"""
    ctx = context or Context()
    if llm_config is not None:
        return ctx.llm_with_cost_manager_from_llm_config(llm_config)
    return ctx.llm()
