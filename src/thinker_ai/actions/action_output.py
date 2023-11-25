import re
from typing import Dict, Type

from pydantic import BaseModel, create_model, root_validator, validator

from thinker_ai.utils.logs import logger
from thinker_ai.utils.output_parser import OutputParser


class ActionOutput:
    content: str
    instruct_content: BaseModel

    def __init__(self, content: str, instruct_content: BaseModel):
        self.content = content
        self.instruct_content = instruct_content


