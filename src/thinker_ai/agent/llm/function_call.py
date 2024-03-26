from typing import Dict

from pydantic import BaseModel


class FunctionCall(BaseModel):
    """The base model for function call"""
    name: str
    arguments: Dict