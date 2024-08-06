from typing import List

from openai.types.beta import AssistantToolParam

from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from thinker_ai.configs.config import config
from thinker_ai_tests.login_test import client


class OpenAiAssistantApiBuilder:
    @staticmethod
    def create_assistant_api(user_id: str, name: str, instructions: str, description: str = None,
                             tools: List[AssistantToolParam] = None, file_ids: List[str] = None,
                             model: str = config.llm.model) -> OpenAiAssistantApi:
        assistant = client.beta.assistants.create(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools,
            file_ids=file_ids,
            description=description)
        assistant_api = OpenAiAssistantApi.from_instance(user_id, assistant)
        return assistant_api
