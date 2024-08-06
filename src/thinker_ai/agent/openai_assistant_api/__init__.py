from typing import cast

from thinker_ai.agent.provider import OpenAILLM
from thinker_ai.agent.provider.llm import LLM

openai_client = cast(OpenAILLM, LLM()).client
