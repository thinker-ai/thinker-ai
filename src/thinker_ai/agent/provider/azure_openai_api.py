
from openai import AsyncAzureOpenAI
from openai._base_client import AsyncHttpxClientWrapper

from thinker_ai.agent.provider.llm_provider_registry import register_provider
from thinker_ai.agent.provider.openai_api import OpenAILLM
from thinker_ai.configs.llm_config import LLMType


@register_provider(LLMType.AZURE)
class AzureOpenAILLM(OpenAILLM):
    """
    Check https://platform.openai.com/examples for examples
    """

    def _init_client(self):
        kwargs = self._make_client_kwargs()
        # https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix
        self.aclient = AsyncAzureOpenAI(**kwargs)
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model

    def _make_client_kwargs(self) -> dict:
        kwargs = dict(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.base_url,
        )

        # to use proxy, openai v1 needs http_client
        proxy_params = self._get_proxy_params()
        if proxy_params:
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs
