import os

from thinker_ai.llm import llm_type, LLM_TYPE
from thinker_ai.llm.claude2_api import Claude2, ClaudeConfig
from thinker_ai.llm.gpt_api import GPT, GPT_Config
from thinker_ai.llm.llm_api import LLM_API

_instances = {}


def get_llm() -> LLM_API:
    # Check if an instance for the given user_id already exists
    if llm_type in _instances:
        return _instances[llm_type]
    if llm_type == LLM_TYPE.OPEN_AI:
        config = GPT_Config(api_key=os.environ.get("OPENAI_API_KEY"),
                            model=os.environ.get("model") or "gpt-4-0613",
                            temperature=os.environ.get("temperature") or 0,
                            max_budget=os.environ.get("max_budget") or 3.0,
                            max_tokens_rsp=os.environ.get("max_tokens_rsp") or 2048,
                            rpm=os.environ.get("rpm") or 10,
                            )
        instance = GPT(config)
    elif llm_type == LLM_TYPE.CLAUDE:
        instance = Claude2(ClaudeConfig(api_key=os.environ.get("ANTHROPIC_API_KEY")))
    else:
        raise f"Unknown LLM_TYPE {llm_type}"
    # Store the created instance in the dictionary and return it
    _instances[llm_type] = instance
    return instance
