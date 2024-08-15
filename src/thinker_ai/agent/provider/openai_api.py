from __future__ import annotations

import asyncio
import json
import re
import os
import time
from typing import Optional, List, Dict, Union, Any, Callable, Type, Literal
from thinker_ai.agent.provider.llm_provider_registry import register_provider
from langchain_core.tools import BaseTool
from openai import OpenAI, AsyncOpenAI, APIConnectionError, AsyncStream, Stream, NOT_GIVEN, NotGiven
from openai._base_client import AsyncHttpxClientWrapper
from openai.pagination import SyncCursorPage
from openai.types import FileObject, ModelDeleted, Image, CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.fine_tuning import FineTuningJob, FineTuningJobEvent
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, after_log, wait_fixed, retry_if_exception_type, wait_random_exponential
from thinker_ai.configs.llm_config import LLMConfig, LLMType
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.agent.provider.constant import GENERAL_FUNCTION_SCHEMA
from thinker_ai.agent.provider.cost_manager import CostManager
from thinker_ai.agent.provider.token_counter import get_openrouter_tokens
from thinker_ai.agent.tools.function_call import FunctionCall
from thinker_ai.agent.provider.llm_schema import PromptMessage
from thinker_ai.agent.tools.openai_callables_register import CallableRegister
from thinker_ai.common.common import log_and_reraise, decode_image
from thinker_ai.common.exceptions import handle_exception
from thinker_ai.common.logs import logger
from thinker_ai.configs.const import USE_CONFIG_TIMEOUT,PROJECT_ROOT
from thinker_ai.configs.llm_config import LLMConfig
from thinker_ai.utils.code_parser import CodeParser
from thinker_ai.utils.text_parser import TextParser


class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""

    def __init__(self, rpm=10):
        self.last_call_time = 0
        # Here is used because even if the calls are made strictly according to time,
        # they will still be QOS'd; consider switching to simple error retry later
        self.interval = 1.1 * 60 / rpm
        self.rpm = rpm

    def split_batches(self, batch: dict[str, PromptMessage]) -> list[dict[str, PromptMessage]]:
        items = list(batch.items())
        return [{k: v for k, v in items[i:i + self.rpm]} for i in range(0, len(items), self.rpm)]

    async def wait_if_needed(self, num_requests):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time

        if elapsed_time < self.interval * num_requests:
            remaining_time = self.interval * num_requests - elapsed_time
            logger.info(f"sleep {remaining_time}")
            await asyncio.sleep(remaining_time)

        self.last_call_time = time.time()


class FunctionException(Exception):
    pass

@register_provider(LLMType.OPENAI)
class OpenAILLM(BaseLLM):
    """
    Check https://platform.openai.com/examples for examples
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        if config.api_key is None or config.api_key=="sk-":
            config.api_key = os.getenv("OPENAI_PROJECT_API_KEY")
        self.client = self.__init_n_openai(config)
        self.aclient = self.__init_a_openai(config)
        self._cost_manager = CostManager()
        self.callables_register=CallableRegister()

    def __init_a_openai(self, config: LLMConfig) -> AsyncOpenAI:
        openai = AsyncOpenAI(api_key=config.api_key)
        return self.__init__openai(config, openai)

    def __init_n_openai(self, config: LLMConfig) -> OpenAI:
        openai = OpenAI(api_key=config.api_key)
        return self.__init__openai(config, openai)

    def __init__openai(self, config: LLMConfig, openai: Union[OpenAI, AsyncOpenAI]) -> Union[OpenAI, AsyncOpenAI]:
        openai.api_base = config.base_url
        openai.proxy = config.proxy
        openai.timeout = self.get_timeout()
        openai.max_retries = config.max_retries
        if config.api_type:
            openai.api_type = config.api_type
            openai.api_version = config.api_version
        return openai


    def get_model(self, model: Optional[str] = None):
        if model:
            return model
        else:
            return self.config.model


    default_system_msg = 'You are a helpful agent.'

    def generate(self, model: str, user_msg: str, system_msg: Optional[str] = None, stream=False) -> str:
        """不感知已注册的function，需要感知已注册的function，使用generate_with_function"""
        if stream:
            return self._chat_completion_stream(PromptMessage(user_msg, system_msg), model)
        else:
            return self._chat_completion(PromptMessage(user_msg, system_msg), model)

    def generate_with_function(self, model: str, user_msg: str, system_msg: Optional[str] = None) -> str:
        """function的同步只支持非流"""
        return self._chat_completion_with_function(PromptMessage(user_msg, system_msg), model)

    async def a_generate(self, user_msg: str, system_msg: Optional[str] = None, model: Optional[str] = None,
                         stream=False) -> str:
        """不感知已注册的function，需要感知已注册的function，使用a_generate_with_function"""
        if stream:
            return await self._a_do_with_normal_stream(PromptMessage(user_msg, system_msg), model)
        else:
            return await self._a_chat_completion(PromptMessage(user_msg, system_msg), model)

    async def a_generate_with_function(self, model: str, user_msg: str, system_msg: Optional[str] = None) -> str:
        """function的异步只支持流"""
        return await self._a_chat_completion_with_function_stream(PromptMessage(user_msg, system_msg), model)

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["agent"] == "agent"])

    def _get_choice_text(self, rsp: ChatCompletion) -> str:
        return rsp.choices[0].message.content

    def _cons_kwargs(self, messages: List[Dict],model:Optional[str]=None, timeout: Optional[int] = None,**extra_kwargs) -> Dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._cost_manager.get_max_tokens(self.get_model(model), messages),
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            # "stop": None,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "timeout": self.get_timeout(timeout),
        }
        # if proxy_params := self._get_proxy_params():
        #     kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    # def _get_proxy_params(self) -> dict:
    #     params = {}
    #     if self.config.proxy:
    #         params = {"proxies": self.config.proxy}
    #         if self.config.base_url:
    #             params["base_url"] = self.config.base_url
    #     return params

    def _chat_completion_with_function(self, prompt: PromptMessage, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
        functions_schema = self.callables_register.callables_schema
        if functions_schema is None or len(self.callables_register.callables_schema) == 0:
            extra_kwargs = {
                "functions": self.callables_register.callables_schema
            }
            kwargs = self._cons_kwargs(messages=prompt.to_dicts(), model=model,timeout=timeout, **extra_kwargs)
            response = self.client.chat.completions.create(model=self.get_model(model), **kwargs)
            if self.config.calc_usage:
                self.cost_manager.update_costs(self.get_model(model), response.usage)
            response = self._do_with_function_call(model=self.get_model(model), prompt=prompt,
                                               rsp_with_function_call=response)
            if self.config.calc_usage:
                self.cost_manager.update_costs(self.get_model(model), response.usage)
            return self._get_choice_text(response)
        else:
            raise FunctionException(f"No functions registered")

    def _chat_completion(self, prompt: PromptMessage, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
        kwargs = self._cons_kwargs(messages=prompt.to_dicts(), model=model,timeout=timeout,)
        response = self.client.chat.completions.create(model=self.get_model(model), **kwargs)
        if self.config.calc_usage:
            self.cost_manager.update_costs(self.get_model(model), response.usage)
        return self._get_choice_text(response)

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        return await self._a_chat_completion(PromptMessage.from_messages(messages), timeout=timeout)

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        return await self._a_do_with_normal_stream(PromptMessage.from_messages(messages))

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT)-> str:
        return await self._achat_completion(messages, timeout=timeout)

    async def _a_chat_completion(self, prompt: PromptMessage, model: Optional[str] = None,
                                 timeout: Optional[int] = None) -> str:
        kwargs = self._cons_kwargs(messages=prompt.to_dicts(), model=model,timeout=timeout,)
        #这个方法会导致debug模式下主程序不能正常结束，处于必须手动中断的状态
        response = await self.aclient.chat.completions.create(model=self.get_model(model), **kwargs)
        if self.config.calc_usage:
            self.cost_manager.update_costs(self.get_model(model), response.usage)
        return self._get_choice_text(response)

    def _chat_completion_stream(self, prompt: PromptMessage, model: Optional[str] = None,
                                 timeout: Optional[int] = None) -> str:
        functions_schema = self.callables_register.callables_schema
        if functions_schema is None or len(self.callables_register.callables_schema) == 0:
            return self._do_with_normal_stream(model=model, prompt=prompt,timeout=timeout)
        else:
            raise FunctionException(
                "The combination of streaming processing and synchronous calls does not apply to functiion_call")

    def _do_with_normal_stream(self, prompt, model: Optional[str]= None, timeout: Optional[int] = None):
        kwargs = self._cons_kwargs(messages=prompt.to_dicts(), model=model,timeout=timeout,)
        stream_rsp = self.client.chat.completions.create(model=self.get_model(model), **kwargs, stream=True)
        full_reply_content,_ = self._extract_from_stream_rsp(stream_rsp)
        self._update_costs_for_stream_rsp(model,prompt,stream_rsp)
        return full_reply_content


    async def _a_do_with_normal_stream(self, prompt, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
        kwargs = self._cons_kwargs(messages=prompt.to_dicts(), model=model,timeout=timeout)
        # 这个方法会导致debug模式下主程序不能正常结束，处于必须手动中断的状态
        response = await self.aclient.chat.completions.create(model=self.get_model(model), **kwargs, stream=True)
        full_reply_content,usage = await self._a_extract_from_stream_rsp(response)
        if self.config.calc_usage:
            if usage is None:
                usage = self._cost_manager.calc_usage(self.get_model(model), prompt.to_dicts(), full_reply_content)
            self.cost_manager.update_costs(self.get_model(model), usage)
        return full_reply_content

    def _extract_from_stream_rsp(self, response: Stream[ChatCompletionChunk]) -> [str,Optional[CompletionUsage]]:
        usage:Optional[CompletionUsage] = None
        collected_messages = []
        for chunk in response:
            usage = self._do_with_chunk(chunk, collected_messages, usage)
        full_reply_content = "".join([m for m in collected_messages])
        return full_reply_content,usage

    async def _a_extract_from_stream_rsp(self, response: AsyncStream[ChatCompletionChunk]) -> [str,Optional[CompletionUsage]]:
        usage:Optional[CompletionUsage] = None
        collected_messages = []
        async for chunk in response:
            usage = self._do_with_chunk(chunk, collected_messages, usage)
        full_reply_content = "".join(collected_messages)
        return full_reply_content,usage

    def _do_with_chunk(self, chunk, collected_messages, usage):
        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
        finish_reason = (
            chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
        )
        collected_messages.append(chunk_message)
        if finish_reason:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                # Some services have usage as an attribute of the chunk, such as Fireworks
                if isinstance(chunk.usage, CompletionUsage):
                    usage = chunk.usage
                else:
                    usage = CompletionUsage(**chunk.usage.dict())
            elif hasattr(chunk.choices[0], "usage"):
                # The usage of some services is an attribute of chunk.choices[0], such as Moonshot
                usage = CompletionUsage(**chunk.choices[0].usage)
        return usage

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        after=after_log(logger, logger.level('WARNING').no),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def _parse_text_to_cls(self, content: str, output_data_mapping: dict) -> Any:
        instruct_content = TextParser.parse_data_with_mapping(content, output_data_mapping)
        return instruct_content

    @staticmethod
    def _parse_function_call(rsp_with_function_call: ChatCompletion) -> FunctionCall:
        function_call = rsp_with_function_call.choices[0].message.function_call
        if function_call is not None:
            return FunctionCall(name=function_call.name, arguments=json.loads(function_call.arguments))

    async def _a_chat_completion_with_function_stream(self, prompt: PromptMessage, model: Optional[str] = None,
                                                      timeout: Optional[int] = None) -> str:
        functions_schema = self.callables_register.callables_schema
        if functions_schema is None or len(self.callables_register.callables_schema) == 0:
            raise FunctionException('function not found')
        extra_kwargs = {
            "functions": functions_schema
        }
        kwargs = self._cons_kwargs(messages=prompt.to_dicts(),
                                   model=model,
                                   timeout=timeout,
                                   **extra_kwargs)
        rsp_with_function_call = await self.aclient.chat.completions.create(model=self.get_model(model), stream=True, **kwargs)
        await self._a_update_costs_for_stream_rsp(model, prompt, rsp_with_function_call)
        function_call = await self._a_parse_function_call_from_steam(rsp_with_function_call)
        if function_call:
            kwargs = self.build_kwargs_after_call_function(function_call, model, prompt, timeout)
            stream_rsp: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
                model=self.get_model(model), **kwargs,
                stream=True)
            await self._a_update_costs_for_stream_rsp(model, prompt, stream_rsp)
            full_reply_content,_ =  await self._a_extract_from_stream_rsp(stream_rsp)
            return full_reply_content

    async def _a_update_costs_for_stream_rsp(self, model, prompt, stream_rsp):
        if self.config.calc_usage:
            full_reply_content,usage = await self._a_extract_from_stream_rsp(stream_rsp)
            if usage is None:
                usage = self._cost_manager.calc_usage(self.get_model(model), prompt.to_dicts(), full_reply_content)
            self._cost_manager.update_costs(self.get_model(model), usage)

    def _update_costs_for_stream_rsp(self, model, prompt, stream_rsp):
        if self.config.calc_usage:
            full_reply_content,usage = self._extract_from_stream_rsp(stream_rsp)
            if usage is None:
                usage = self._cost_manager.calc_usage(self.get_model(model), prompt.to_dicts(), full_reply_content)
            self._cost_manager.update_costs(self.get_model(model), usage)

    def _do_with_function_call(self, prompt: PromptMessage,
                               rsp_with_function_call: ChatCompletion, model: Optional[str] = None,
                               timeout: Optional[int] = None) -> ChatCompletion:
        function_call = self._parse_function_call(rsp_with_function_call)
        if function_call:
            kwargs = self.build_kwargs_after_call_function(function_call, model, prompt, timeout)
            return self.client.chat.completions.create(model=self.get_model(model), **kwargs)

    async def _a_do_with_function_call(self, prompt: PromptMessage,
                                       rsp_with_function_call: AsyncStream[ChatCompletionChunk],
                                       model: Optional[str] = None, timeout: Optional[int] = None) -> ChatCompletion:
        function_call = await self._a_parse_function_call_from_steam(rsp_with_function_call)
        if function_call:
            kwargs = self.build_kwargs_after_call_function(function_call, self.get_model(model), prompt, timeout)
            return await self.aclient.chat.completions.create(**kwargs)

    def build_kwargs_after_call_function(self, function_call, model, prompt, timeout):
        result = self._call_function(function_call)
        usg_msg = self._build_user_message_for_function_result(prompt, result)
        kwargs = self._cons_kwargs(messages=PromptMessage(usg_msg).to_dicts(),model=model,timeout=timeout)
        return kwargs

    async def _a_parse_function_call_from_steam(self, rsp_with_function_call: AsyncStream[
        ChatCompletionChunk]) -> FunctionCall:
        name = None
        collected_arguments = []
        # iterate through the stream of events
        async for chunk in rsp_with_function_call:
            function_call = chunk.choices[0].delta.function_call  # extract the message
            if function_call is not None:
                if name is None:
                    name = function_call.name
                collected_arguments.append(function_call.arguments)  # save the message
        print()
        arguments: str = "".join([m for m in collected_arguments])
        return FunctionCall(name=name, arguments=json.loads(arguments))

    def _call_function(self, function_call: FunctionCall) -> Any:
        function = self.callables_register.get_langchain_tool(function_call.name)
        # 调用函数
        return function.invoke(function_call.arguments)

    def _build_user_message_for_function_result(self, prompt: PromptMessage, function_result: str) -> str:
        result_prompt_template = """
           用户提出的原始问题是“{user_message}”,原始的系统提示是{system_message},现在已经有了数据:“{function_result}”，请用这个数据提供自然语言的回复
        """
        return result_prompt_template.format(user_message=prompt.user_message.content,
                                             system_message=prompt.system_message.content,
                                             function_result=function_result)

    def register_langchain_function(self, tool: BaseTool):
        self.callables_register.register_langchain_tool(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.callables_register.register_callable(func, args_schema)

    def register_langchain_functions(self, tool_names: list[str]):
        self.callables_register.register_langchain_tool_names(tool_names)

    def remove_function(self, name: str):
        self.callables_register.remove_callable(name)

    async def a_generate_batch(self, model: str, user_msgs: dict[str, str], sys_msg: Optional[str] = None) -> dict[
        str, str]:
        """仅返回纯文本"""
        messages: dict[str, ChatCompletion] = await self._a_completion_batch(user_msgs, sys_msg, model)
        results: dict[str, str] = {key: self._get_choice_text(message) for key, message in messages.items()}
        return results

    async def _a_completion_batch(self, user_msgs: dict[str, str], sys_msg: Optional[str] = None,
                                  model: Optional[str] = None) -> dict[
        str, ChatCompletion]:
        rate_limiter = RateLimiter(rpm=self.config.rpm)
        batch_prompt = {key: PromptMessage(user_msg, sys_msg) for key, user_msg in user_msgs.items()}
        split_batches: list[dict[str, PromptMessage]] = rate_limiter.split_batches(batch_prompt)
        all_results: dict[str, ChatCompletion] = {}
        for small_batch in split_batches:
            await rate_limiter.wait_if_needed(len(small_batch))
            future_map = {
                key: self._a_chat_completion(prompt, model) for
                key, prompt in small_batch.items()}
            # Gather the results of these futures
            results = await asyncio.gather(*future_map.values())
            # Map the results back to their respective keys
            for key, result in zip(future_map.keys(), results):
                all_results[key] = result
        return all_results

    def upload_file(self, file_dir: str, purpose: Literal["fine-tune", "assistants"]) -> FileObject:
        file = self.client.files.create(
            file=open(file_dir, "rb"),
            purpose=purpose
        )
        return file

    def delete_file(self, file_id: str) -> bool:
        deleted = self.client.files.delete(file_id)
        return deleted.deleted

    def create_fine_tuning(self, training_file_id: str,
                           hyperparameters: Hyperparameters | NotGiven = NOT_GIVEN,
                           model: Optional[str] = None) -> FineTuningJob:
        kwargs = {
            "training_file": training_file_id,
            "model": self.get_model(model),
        }
        # 如果hyperparameters被明确设置，添加到kwargs字典中
        if hyperparameters is not NOT_GIVEN:
            kwargs["hyperparameters"] = hyperparameters
        # 使用动态参数字典调用方法
        job = self.client.fine_tuning.jobs.create(**kwargs)
        return job

    def List_fine_tuning_jobs(self, limit=10) -> SyncCursorPage[FineTuningJob]:
        return self.client.fine_tuning.jobs.list(limit=limit)

    def retrieve_tuning_job(self, training_file_id: str) -> FineTuningJob:
        # Retrieve the state of a fine-tune
        return self.client.fine_tuning.jobs.retrieve(training_file_id)

    def cancel_tuning_job(self, training_file_id: str) -> FineTuningJob:
        # Cancel a job
        return self.client.fine_tuning.jobs.cancel(training_file_id)

    def list_events_from_fine_tuning_job(self, training_file_id: str, limit=10) -> SyncCursorPage[FineTuningJobEvent]:
        # list_events_from_fine_tuning_job
        return self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=training_file_id, limit=limit)

    def delete_tuning_job(self, fine_tuned_model_id: str) -> bool:
        # Delete a fine-tuned model (must be an owner of the org the model was created in)
        modelDeleted: ModelDeleted = self.client.models.delete(fine_tuned_model_id)
        return modelDeleted.deleted

    @handle_exception
    async def amoderation(self, content: Union[str, list[str]]):
        """Moderate content."""
        return await self.aclient.moderations.create(input=content)

    async def atext_to_speech(self, **kwargs):
        """text to speech"""
        return await self.aclient.audio.speech.create(**kwargs)

    async def aspeech_to_text(self, **kwargs):
        """speech to text"""
        return await self.aclient.audio.transcriptions.create(**kwargs)

    async def gen_image(
        self,
        prompt: str,
        size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        model: str = None,
        resp_format: Literal["url", "b64_json"] = "url",
    ) -> list["Image"]:
        """image generate"""
        assert resp_format in ["url", "b64_json"]
        if not model:
            model = self.model
        res = await self.aclient.images.generate(
            model=model, prompt=prompt, size=size, quality=quality, n=1, response_format=resp_format
        )
        imgs = []
        for item in res.data:
            img_url_or_b64 = item.url if resp_format == "url" else item.b64_json
            imgs.append(decode_image(img_url_or_b64))
        return imgs

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        after=after_log(logger, logger.level("WARNING").no),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout)
        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout))
        return rsp

    async def _achat_completion_function(
        self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT, **chat_configs
    ) -> ChatCompletion:
        messages = self.format_msg(messages)
        kwargs = self._cons_kwargs(messages=messages,timeout=self.get_timeout(timeout), **chat_configs)
        rsp: ChatCompletion = await self.aclient.chat.completions.create(self.get_model(), **kwargs)
        self.cost_manager.update_costs(self.pricing_plan,rsp.usage)
        return rsp

    async def aask_code(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT, **kwargs) -> dict:
        """Use function of tools to ask a code.
        Note: Keep kwargs consistent with https://platform.openai.com/docs/api-reference/chat/create

        Examples:
             llm = OpenAILLM()
             msg = [{'role': 'user', 'content': "Write a python hello world code."}]
             rsp = await llm.aask_code(msg)
        # -> {'language': 'python', 'code': "print('Hello, World!')"}
        """
        if "tools" not in kwargs:
            configs = {"tools": [{"type": "function", "function": GENERAL_FUNCTION_SCHEMA}]}
            kwargs.update(configs)
        rsp = await self._achat_completion_function(messages, **kwargs)
        return self.get_choice_function_arguments(rsp)

    def _parse_arguments(self, arguments: str) -> dict:
        """parse arguments in openai function call"""
        if "language" not in arguments and "code" not in arguments:
            logger.warning(f"Not found `code`, `language`, We assume it is pure code:\n {arguments}\n. ")
            return {"language": "python", "code": arguments}

        # 匹配language
        language_pattern = re.compile(r'[\"\']?language[\"\']?\s*:\s*["\']([^"\']+?)["\']', re.DOTALL)
        language_match = language_pattern.search(arguments)
        language_value = language_match.group(1) if language_match else "python"

        # 匹配code
        code_pattern = r'(["\'`]{3}|["\'`])([\s\S]*?)\1'
        try:
            code_value = re.findall(code_pattern, arguments)[-1][-1]
        except Exception as e:
            logger.error(f"{e}, when re.findall({code_pattern}, {arguments})")
            code_value = None

        if code_value is None:
            raise ValueError(f"Parse code error for {arguments}")
        # arguments只有code的情况
        return {"language": language_value, "code": code_value}

    # @handle_exception
    def get_choice_function_arguments(self, rsp: ChatCompletion) -> dict:
        """Required to provide the first function arguments of choice.

        :param dict rsp: same as in self.get_choice_function(rsp)
        :return dict: return the first function arguments of choice, for example,
            {'language': 'python', 'code': "print('Hello, World!')"}
        """
        message = rsp.choices[0].message
        if (
            message.tool_calls is not None
            and message.tool_calls[0].function is not None
            and message.tool_calls[0].function.arguments is not None
        ):
            # reponse is code
            try:
                return json.loads(message.tool_calls[0].function.arguments, strict=False)
            except json.decoder.JSONDecodeError as e:
                error_msg = (
                    f"Got JSONDecodeError for \n{'--'*40} \n{message.tool_calls[0].function.arguments}, {str(e)}"
                )
                logger.error(error_msg)
                return self._parse_arguments(message.tool_calls[0].function.arguments)
        elif message.tool_calls is None and message.content is not None:
            # reponse is code, fix openai tools_call respond bug,
            # The response content is `code``, but it appears in the content instead of the arguments.
            code_formats = "```"
            if message.content.startswith(code_formats) and message.content.endswith(code_formats):
                code = CodeParser.parse_code(None, message.content)
                return {"language": "python", "code": code}
            # reponse is message
            return {"language": "markdown", "code": self.get_choice_text(rsp.dict())}
        else:
            logger.error(f"Failed to parse \n {rsp}\n")
            raise Exception(f"Failed to parse \n {rsp}\n")