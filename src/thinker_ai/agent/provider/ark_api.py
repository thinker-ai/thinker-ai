from openai import AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from thinker_ai.agent.provider.llm_provider_registry import register_provider
from thinker_ai.agent.provider.openai_api import OpenAILLM
from thinker_ai.common.logs import log_llm_stream
from thinker_ai.configs.const import USE_CONFIG_TIMEOUT
from thinker_ai.configs.llm_config import LLMType


@register_provider(LLMType.ARK)
class ArkLLM(OpenAILLM):
    """
    用于火山方舟的API
    见：https://www.volcengine.com/docs/82379/1263482
    """

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout)),
            stream=True,
            extra_body={"stream_options": {"include_usage": True}}  # 只有增加这个参数才会在流式时最后返回usage
        )
        usage = None
        collected_messages = []
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
            log_llm_stream(chunk_message)
            collected_messages.append(chunk_message)
            if chunk.usage:
                # 火山方舟的流式调用会在最后一个chunk中返回usage,最后一个chunk的choices为[]
                usage = CompletionUsage(**chunk.usage)

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)
        self.cost_manager.update_costs(self.get_model(),usage)
        return full_reply_content

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self.cost_manager.update_costs(self.get_model(rsp.model),rsp.usage)
        return rsp
