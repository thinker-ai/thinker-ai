from enum import Enum
from typing import List

from openai.types.beta import AssistantToolParam

from thinker_ai.agent.assistant_api import AssistantApi
from thinker_ai.agent.openai_assistant_api.openai_assistant_api_builder import OpenAiAssistantApiBuilder
from thinker_ai.configs.config import config
from thinker_ai.configs.llm_config import LLMType


class AssistantApiBuilder:
    @staticmethod
    def create(name: str, instructions: str, description: str = None,
               tools: List[AssistantToolParam] = None,
               model: str = config.llm.model) -> AssistantApi:
        if config.llm.api_type == LLMType.OPENAI:
            assistant_api = OpenAiAssistantApiBuilder.create(
                name=name,
                instructions=instructions,
                description=description,
                tools=tools,
                model=model)
            return assistant_api

    @staticmethod
    def retrieve(assistant_id: str) -> AssistantApi:
        if config.llm.api_type == LLMType.OPENAI:
            assistant_api = OpenAiAssistantApiBuilder.retrieve(assistant_id=assistant_id)
            return assistant_api
