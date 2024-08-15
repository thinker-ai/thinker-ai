from typing import Optional
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.configs.llm_config import LLMConfig
from thinker_ai.app_context import AppContext


def LLM(llm_config: Optional[LLMConfig] = None, context: AppContext = None) -> BaseLLM:
    """get the default llm provider if name is None"""
    ctx = context or AppContext()
    if llm_config is not None:
        return ctx.llm_with_cost_manager_from_llm_config(llm_config)
    return ctx.llm()
