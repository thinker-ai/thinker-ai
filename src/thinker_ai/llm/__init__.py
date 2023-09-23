import os
from enum import Enum

import requests
import openai

openai.requestssession = requests.Session
if os.environ.get("HTTP_PROXY"):
   openai.proxy = os.environ.get("HTTP_PROXY")
class LLM_TYPE(Enum):
    OPEN_AI = 1
    CLAUDE = 2

    @classmethod
    def type_of(cls, type_str: str) -> 'LLM_TYPE':
        try:
            return LLM_TYPE[type_str]
        except KeyError:
            raise ValueError(f"'{type_str}' is not a valid LLM_TYPE.")

if os.environ.get("LLM_TYPE"):
    llm_type = LLM_TYPE.type_of(os.environ.get("LLM_TYPE"))
else:
    raise Exception("System environment variable missing: LLM_TYPE")
