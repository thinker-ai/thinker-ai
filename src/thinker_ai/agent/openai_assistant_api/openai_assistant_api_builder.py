from typing import List

from openai.types.beta import AssistantToolParam

from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from thinker_ai.configs.config import config


class OpenAiAssistantApiBuilder:
    @staticmethod
    def create(name: str, instructions: str, description: str = None,
               tools: List[AssistantToolParam] = None,
               model: str = config.llm.model) -> OpenAiAssistantApi:
        assistant = openai.client.beta.assistants.create(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools,
            description=description)
        assistant_api = OpenAiAssistantApi.from_instance(assistant)
        return assistant_api

    @staticmethod
    def retrieve(assistant_id: str) -> OpenAiAssistantApi:
        assistant = openai.client.beta.assistants.retrieve(assistant_id)
        assistant_api = OpenAiAssistantApi.from_instance(assistant)
        return assistant_api
