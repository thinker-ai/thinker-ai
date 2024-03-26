from typing import Dict

from pydantic import BaseModel

from thinker_ai.utils.serializable import Serializable


class FunctionCall(Serializable):
    """The base model for function call"""
    name: str
    arguments: Dict