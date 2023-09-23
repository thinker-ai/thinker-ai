import os

import openai
import requests

from thinker_ai.config import configs
from thinker_ai.llm import llm_type
from thinker_ai.llm.claude2 import Claude2, ClaudeConfig
from thinker_ai.llm.gpt import GPT, GPT_Config
from thinker_ai.llm.llm_api import LLM_API, LLM_TYPE




_instances = {}

def get_llm() -> LLM_API:
    # Check if an instance for the given user_id already exists
    if llm_type in _instances:
        return _instances[llm_type]
    if llm_type == LLM_TYPE.OPEN_AI:
        instance = GPT(GPT_Config(api_key=configs.get("OPENAI_API_KEY")))
    elif llm_type == LLM_TYPE.CLAUDE:
        instance = Claude2(ClaudeConfig(api_key=configs.get("ANTHROPIC_API_KEY")))
    else:
        raise f"Unknown LLM_TYPE {llm_type}"
    # Store the created instance in the dictionary and return it
    _instances[llm_type] = instance
    return instance
