from typing import cast

from thinker_ai.agent.provider import OpenAILLM
from thinker_ai.context_mixin import ContextMixin

client = (cast(OpenAILLM, ContextMixin().llm)).client
