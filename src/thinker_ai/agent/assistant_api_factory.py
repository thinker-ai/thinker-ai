from enum import Enum
from typing import List, Literal

from openai.types.beta import AssistantToolParam

from thinker_ai.agent.assistant_api import AssistantApi
from thinker_ai.agent.openai_assistant_api.openai_assistant_api_builder import OpenAiAssistantApiBuilder
from thinker_ai.configs.config import config


class PROVIDER(Enum):
    GeminiLLM = "GeminiLLM"
    OpenAILLM = "OpenAILLM"
    AzureOpenAILLM = "AzureOpenAILLM"
    OllamaLLM = "OllamaLLM"
    SparkLLM = "SparkLLM"
    QianFanLLM = "QianFanLLM"
    DashScopeLLM = "DashScopeLLM"
    AnthropicLLM = "AnthropicLLM"
    BedrockLLM = "BedrockLLM"
    ArkLLM = "ArkLLM"


class AssistantApiFactory:
    @staticmethod
    def create(user_id: str, name: str, instructions: str, description: str = None,
               tools: List[AssistantToolParam] = None, file_ids: List[str] = None,
               provider: PROVIDER = PROVIDER.OpenAILLM,
               model: str = config.llm.model) -> AssistantApi:
        if provider == PROVIDER.OpenAILLM:
            return OpenAiAssistantApiBuilder.create_assistant_api(
                user_id=user_id,
                name=name,
                instructions=instructions,
                description=description,
                tools=tools,
                file_ids=file_ids,
                model=model)
