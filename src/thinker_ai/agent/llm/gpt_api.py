from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, List, Dict, Union, Any, Callable, Type, Literal

from langchain_core.tools import BaseTool
from openai import OpenAI, AsyncOpenAI, APIConnectionError, AsyncStream, Stream, NOT_GIVEN, NotGiven
from openai.pagination import SyncCursorPage
from openai.types import FileObject, ModelDeleted
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.fine_tuning import FineTuningJob, FineTuningJobEvent
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, after_log, wait_fixed, retry_if_exception_type

from thinker_ai.agent.functions.functions_register import FunctionsRegister
from thinker_ai.agent.llm.cost_manager import CostManager
from thinker_ai.agent.llm.function_call import FunctionCall
from thinker_ai.agent.llm.gpt_schema import PromptMessage
from thinker_ai.utils.text_parser import TextParser
from thinker_ai.utils.logs import logger


class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""

    def __init__(self, rpm):
        self.last_call_time = 0
        # Here 1.1 is used because even if the calls are made strictly according to time,
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


def log_and_reraise(retry_state):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    raise retry_state.outcome.exception()


class GPT_Config(BaseModel):
    type: str = None
    version: str = None
    api_key: str
    temperature: int = 0
    max_budget: float = 3.0
    auto_max_tokens: bool = False
    max_tokens_rsp: int = 2048
    proxy: str = None
    api_base: str = "https://api.openai.com/v1"
    rpm: int = 10
    timeout: float = 3.0
    max_retries: int = 3


class FunctionException(Exception):
    pass


class GPT:
    """
    Check https://platform.openai.com/examples for examples
    """

    def __init__(self, config: GPT_Config):
        self.llm = self.__init_n_openai(config)
        self.a_llm = self.__init_a_openai(config)
        self._cost_manager = CostManager(config.max_budget, config.auto_max_tokens, config.max_tokens_rsp)
        self.temperature = config.temperature
        self.rateLimiter = RateLimiter(rpm=config.rpm)
        self.functions_register = FunctionsRegister()

    def __init_a_openai(self, config: GPT_Config) -> AsyncOpenAI:
        openai = AsyncOpenAI(api_key=config.api_key)
        return self.__init__openai(config, openai)

    def __init_n_openai(self, config: GPT_Config) -> OpenAI:
        openai = OpenAI(api_key=config.api_key)
        return self.__init__openai(config, openai)

    def __init__openai(self, config, openai: Union[OpenAI, AsyncOpenAI]) -> Union[OpenAI, AsyncOpenAI]:
        openai.api_base = config.api_base
        openai.proxy = config.proxy
        openai.timeout = config.timeout
        openai.max_retries = config.max_retries
        if config.type:
            openai.api_type = config.type
            openai.api_version = config.version
        return openai

    default_system_msg = 'You are a helpful agent.'

    def generate(self, model: str, user_msg: str, system_msg: Optional[str] = None, stream=False) -> str:
        if stream:
            return self._chat_completion_stream(model, PromptMessage(user_msg, system_msg))
        else:
            output = self._chat_completion(model, PromptMessage(user_msg, system_msg))
            return self._get_choice_text(output)

    async def a_generate(self, model: str, user_msg: str, system_msg: Optional[str] = None, stream=False) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._a_chat_completion_stream(model, PromptMessage(user_msg, system_msg))
        else:
            output = await self._a_chat_completion(model, PromptMessage(user_msg, system_msg))
            return self._get_choice_text(output)

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["agent"] == "agent"])

    def _get_choice_text(self, rsp: ChatCompletion) -> str:
        return rsp.choices[0].message.content

    def _cons_kwargs(self, model: str, messages: List[Dict], timeout: float,
                     candidate_functions: Optional[List[Dict]] = None) -> Dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._cost_manager.get_max_tokens(model, messages),
            "n": 1,
            "stop": None,
            "temperature": self.temperature,
        }
        if candidate_functions is not None and len(candidate_functions) != 0:
            kwargs["functions"] = candidate_functions
        kwargs["timeout"] = timeout
        return kwargs

    def _chat_completion(self, model: str, prompt: PromptMessage) -> ChatCompletion:
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout,
                                   self.functions_register.functions_schema)
        response = self.llm.chat.completions.create(model=model, **kwargs)
        self._cost_manager.update_costs(model, response.usage.dict())
        if kwargs.get("functions") is not None:
            return self._do_with_function_call(model=model, prompt=prompt, rsp_with_function_call=response)
        else:
            return response

    async def _a_chat_completion(self, model: str, prompt: PromptMessage) -> ChatCompletion:
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout,
                                   self.functions_register.functions_schema)
        response = await self.a_llm.chat.completions.create(model=model, **kwargs)
        self._cost_manager.update_costs(model, response.usage.dict())
        if kwargs.get("functions") is not None:
            raise FunctionException(
                "The combination of non-streaming processing and asynchronous calls does not apply to functiion_call")
        else:
            return response

    def _chat_completion_stream(self, model: str, prompt: PromptMessage) -> str:
        functions_schema = self.functions_register.functions_schema
        if functions_schema is None or len(self.functions_register.functions_schema) == 0:
            return self._do_with_normal_stream(model=model, prompt=prompt)
        else:
            raise FunctionException(
                "The combination of streaming processing and synchronous calls does not apply to functiion_call")

    async def _a_chat_completion_stream(self, model: str, prompt: PromptMessage) -> str:
        functions_schema = self.functions_register.functions_schema
        if functions_schema is None or len(self.functions_register.functions_schema) == 0:
            return await self._a_do_with_normal_stream(model, prompt)
        else:
            return await self._a_do_with_function_call_stream(model=model, prompt=prompt)

    def _do_with_normal_stream(self, model, prompt):
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout)
        response = self.llm.chat.completions.create(model=model, **kwargs, stream=True)
        full_reply_content = self._extract_from_stream_rsp(response)
        usage = self._cost_manager.calc_usage(model, prompt.to_dicts(), full_reply_content)
        self._cost_manager.update_costs(model, usage)
        return full_reply_content

    async def _a_do_with_normal_stream(self, model, prompt) -> str:
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout)
        response = await self.a_llm.chat.completions.create(model=model, **kwargs, stream=True)
        full_reply_content = await self._a_extract_from_stream_rsp(response)
        usage = self._cost_manager.calc_usage(model, prompt.to_dicts(), full_reply_content)
        self._cost_manager.update_costs(model, usage)
        return full_reply_content

    def _extract_from_stream_rsp(self, response: Stream[ChatCompletionChunk]) -> str:
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            content = chunk.choices[0].delta.content  # extract the message
            if content is not None:
                print(content, end="")
                collected_messages.append(content)  # save the message
        print()
        full_reply_content = "".join([m for m in collected_messages])
        return full_reply_content

    async def _a_extract_from_stream_rsp(self, response: AsyncStream[ChatCompletionChunk]) -> str:
        # create variables to collect the stream of chunks
        collected_messages = []
        # iterate through the stream of events
        async for chunk in response:
            content = chunk.choices[0].delta.content  # extract the message
            if content is not None:
                print(content, end="")
                collected_messages.append(content)  # save the message
        print()
        full_reply_content = "".join([m for m in collected_messages])
        return full_reply_content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        after=after_log(logger, logger.level('WARNING').name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def _parse_text_to_cls(self, content: str, output_data_mapping: dict, ) -> Any:
        instruct_content = TextParser.parse_data_with_mapping(content, output_data_mapping)
        return instruct_content

    @staticmethod
    def _parse_function_call(rsp_with_function_call: ChatCompletion) -> FunctionCall:
        function_call = rsp_with_function_call.choices[0].message.function_call
        if function_call is not None:
            return FunctionCall(name=function_call.name, arguments=json.loads(function_call.arguments))

    async def _a_do_with_function_call_stream(self, model: str, prompt: PromptMessage) -> str:
        kwargs = self._cons_kwargs(model=model,
                                   messages=prompt.to_dicts(),
                                   timeout=self.a_llm.timeout,
                                   candidate_functions=self.functions_register.functions_schema)
        rsp_with_function_call = await self.a_llm.chat.completions.create(model=model, **kwargs, stream=True)
        function_call = await self._a_parse_function_call_from_steam(rsp_with_function_call)
        if function_call:
            result = self._call_function(function_call)
            usg_msg = self._build_user_message_for_function_result(model, prompt, result)
            kwargs = self._cons_kwargs(model, PromptMessage(usg_msg).to_dicts(), self.a_llm.timeout)
            response: AsyncStream[ChatCompletionChunk] = await self.a_llm.chat.completions.create(model=model, **kwargs,
                                                                                                  stream=True)
            return await self._a_extract_from_stream_rsp(response)

    def _do_with_function_call(self, model: str, prompt: PromptMessage,
                               rsp_with_function_call: ChatCompletion) -> ChatCompletion:
        function_call = self._parse_function_call(rsp_with_function_call)
        if function_call:
            result = self._call_function(function_call)
            usg_msg = self._build_user_message_for_function_result(model, prompt, result)
            kwargs = self._cons_kwargs(model, PromptMessage(usg_msg).to_dicts(), self.a_llm.timeout)
            return self.llm.chat.completions.create(model=model, **kwargs)

    async def _a_do_with_function_call(self, model: str, prompt: PromptMessage,
                                       rsp_with_function_call: AsyncStream[ChatCompletionChunk]) -> ChatCompletion:
        function_call = await self._a_parse_function_call_from_steam(rsp_with_function_call)
        if function_call:
            result = self._call_function(function_call)
            usg_msg = self._build_user_message_for_function_result(model, prompt, result)
            kwargs = self._cons_kwargs(model, PromptMessage(usg_msg).to_dicts(), self.a_llm.timeout)
            return await self.a_llm.chat.completions.create(model=model, **kwargs)

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
        function = self.functions_register.get_function(function_call.name)
        # 调用函数
        return function.invoke(function_call.arguments)

    def _build_user_message_for_function_result(self, model: str, prompt: PromptMessage, function_result: str) -> str:
        result_prompt_template = """
           用户提出的原始问题是“{user_message}”,原始的系统提示是{system_message},现在已经有了数据:“{function_result}”，请用这个数据提供自然语言的回复
        """
        return result_prompt_template.format(user_message=prompt.user_message.content,
                                             system_message=prompt.system_message.content,
                                             function_result=function_result)

    def register_langchain_function(self, tool: BaseTool):
        self.functions_register.register_langchain_tool(tool)

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.functions_register.register_function(func, args_schema)

    def register_langchain_functions(self, tool_names: list[str]):
        self.functions_register.register_langchain_tool_names(tool_names)

    def remove_function(self, name: str):
        self.functions_register.remove_function(name)

    async def a_generate_batch(self, model: str, user_msgs: dict[str, str], sys_msg: Optional[str] = None) -> dict[
        str, str]:
        """仅返回纯文本"""
        messages: dict[str, ChatCompletion] = await self._a_completion_batch(model, user_msgs, sys_msg)
        results: dict[str, str] = {key: self._get_choice_text(message) for key, message in messages.items()}
        return results

    async def _a_completion_batch(self, model: str, user_msgs: dict[str, str], sys_msg: Optional[str] = None) -> dict[
        str, ChatCompletion]:
        batch_prompt = {key: PromptMessage(user_msg, sys_msg) for key, user_msg in user_msgs.items()}
        split_batches: list[dict[str, PromptMessage]] = self.rateLimiter.split_batches(batch_prompt)
        all_results: dict[str, ChatCompletion] = {}
        for small_batch in split_batches:
            await self.rateLimiter.wait_if_needed(len(small_batch))
            future_map = {
                key: self._a_chat_completion(model, prompt) for
                key, prompt in small_batch.items()}
            # Gather the results of these futures
            results = await asyncio.gather(*future_map.values())
            # Map the results back to their respective keys
            for key, result in zip(future_map.keys(), results):
                all_results[key] = result
        return all_results

    def upload_file(self, file_dir: str, purpose: Literal["fine-tune", "assistants"]) -> FileObject:
        file = self.llm.files.create(
            file=open(file_dir, "rb"),
            purpose=purpose
        )
        return file

    def delete_file(self, file_id: str) -> bool:
        deleted = self.llm.files.delete(file_id)
        return deleted.deleted

    def create_fine_tuning(self, model: str, training_file_id: str,hyperparameters:Hyperparameters | NotGiven = NOT_GIVEN) -> FineTuningJob:
        kwargs = {
            "training_file": training_file_id,
            "model": model,
        }
        # 如果hyperparameters被明确设置，添加到kwargs字典中
        if hyperparameters is not NOT_GIVEN:
            kwargs["hyperparameters"] = hyperparameters
        # 使用动态参数字典调用方法
        job = self.llm.fine_tuning.jobs.create(**kwargs)
        return job
    def List_fine_tuning_jobs(self, limit=10) -> SyncCursorPage[FineTuningJob]:
        return self.llm.fine_tuning.jobs.list(limit=limit)

    def retrieve_tuning_job(self, training_file_id: str) -> FineTuningJob:
        # Retrieve the state of a fine-tune
        return self.llm.fine_tuning.jobs.retrieve(training_file_id)

    def cancel_tuning_job(self, training_file_id: str) -> FineTuningJob:
        # Cancel a job
        return self.llm.fine_tuning.jobs.cancel(training_file_id)

    def list_events_from_fine_tuning_job(self, training_file_id: str, limit=10) -> SyncCursorPage[FineTuningJobEvent]:
        # list_events_from_fine_tuning_job
        return self.llm.fine_tuning.jobs.list_events(fine_tuning_job_id=training_file_id, limit=limit)

    def delete_tuning_job(self, fine_tuned_model_id: str) -> bool:
        # Delete a fine-tuned model (must be an owner of the org the model was created in)
        modelDeleted: ModelDeleted = self.llm.models.delete(fine_tuned_model_id)
        return modelDeleted.deleted
