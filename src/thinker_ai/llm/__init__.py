from enum import Enum

import requests
import openai

from thinker_ai.config import configs

def session_factory():
    return requests.Session()

openai.requestssession = session_factory
openai.proxy = configs.get("HTTP_PROXY")
class LLM_TYPE(Enum):
    OPEN_AI = 1
    CLAUDE = 2

    @classmethod
    def type_of(cls, type_str: str) -> 'LLM_TYPE':
        try:
            return LLM_TYPE[type_str]
        except KeyError:
            raise ValueError(f"'{type_str}' is not a valid LLM_TYPE.")

llm_type = LLM_TYPE.type_of(configs.get("llm_type"))