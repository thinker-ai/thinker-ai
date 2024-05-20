from typing import Dict

from thinker_ai.common.serializable import Serializable


class FunctionCall(Serializable):
    """The base model for function call"""
    name: str
    arguments: Dict
