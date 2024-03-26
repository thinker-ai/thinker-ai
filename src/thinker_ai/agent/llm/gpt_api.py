import asyncio
import json
import time
from typing import Optional, List, Dict, Union, Any

from openai import OpenAI, AsyncOpenAI, APIConnectionError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, after_log, wait_fixed, retry_if_exception_type

from thinker_ai.agent.actions.result_parser import ResultParser
from thinker_ai.agent.llm.cost_manager import CostManager
from thinker_ai.agent.llm.function_call import FunctionCall
from thinker_ai.agent.llm.gpt_schema import PromptMessage
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

    def generate(self, model: str, user_msg: str, system_msg: Optional[str] = None) -> str:
        output = self._chat_completion(model, user_msg, system_msg)
        return self._get_choice_text(output)

    async def a_generate(self, model: str, user_msg: str, system_msg: Optional[str] = None, stream=False) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._a_chat_completion_stream(model, user_msg, system_msg)
        rsp = await self._a_chat_completion(model, user_msg, system_msg)
        return self._get_choice_text(rsp)

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["agent"] == "agent"])

    def _get_choice_text(self, rsp: dict) -> str:
        return rsp.get("choices")[0]["message"]["content"]

    async def _a_chat_completion_stream(self, model: str, user_msg: str, system_msg: Optional[str] = None) -> str:
        prompt = PromptMessage(user_msg, system_msg)
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout)
        response = await self.a_llm.chat.completions.create(model=model, **kwargs, stream=True)
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        async for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            content = chunk.choices[0].delta.content  # extract the message
            if content is not None:
                print(content, end="")
                collected_messages.append(content)  # save the message
        print()

        full_reply_content = "".join([m for m in collected_messages])
        usage = self._cost_manager.calc_usage(model, prompt.to_dicts(), full_reply_content)
        self._cost_manager.update_costs(model, usage)
        return full_reply_content

    def _cons_kwargs(self, model: str, messages: list[dict], timeout: float,
                     candidate_functions: Optional[List[Dict]] = None) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._cost_manager.get_max_tokens(model, messages),
            "n": 1,
            "stop": None,
            "temperature": self.temperature,
        }
        if candidate_functions:
            kwargs["functions"] = candidate_functions
        kwargs["timeout"] = timeout
        return kwargs

    def _chat_completion(self, model: str, user_msg: str, system_msg: Optional[str] = None,
                         candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        kwargs = self._cons_kwargs(model, prompt.to_dicts(), self.a_llm.timeout,candidate_functions)
        rsp = self.llm.chat.completions.create(model=model, **kwargs)
        self._cost_manager.update_costs(model, rsp.usage.dict())
        return rsp.dict()

    async def _a_chat_completion(self, model: str, user_msg: str, system_msg: Optional[str] = None,
                                 candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        kwargs = self._cons_kwargs(model, prompt.to_dicts(),self.a_llm.timeout, candidate_functions)
        rsp = await self.a_llm.chat.completions.create(model=model,**kwargs)
        self._cost_manager.update_costs(model, rsp.usage.dict())
        return rsp.dict()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        after=after_log(logger, logger.level('WARNING').name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def a_generate_batch(self, model: str, user_msgs: dict[str, str], sys_msg: Optional[str] = None) -> dict[
        str, str]:
        """仅返回纯文本"""
        messages: dict[str, dict] = await self._a_completion_batch(model, user_msgs, sys_msg)
        results: dict[str, str] = {key: self._get_choice_text(message) for key, message in messages.items()}
        return results

    async def _a_completion_batch(self, model: str, user_msgs: dict[str, str], sys_msg: Optional[str] = None) -> dict[
        str, dict]:
        batch_prompt = {key: PromptMessage(user_msg, sys_msg) for key, user_msg in user_msgs.items()}
        split_batches: list[dict[str, PromptMessage]] = self.rateLimiter.split_batches(batch_prompt)
        all_results = {}
        for small_batch in split_batches:
            await self.rateLimiter.wait_if_needed(len(small_batch))
            future_map = {
                key: self._a_chat_completion(model, prompt.user_message_content, prompt.system_message_content) for
                key, prompt in small_batch.items()}
            # Gather the results of these futures
            results = await asyncio.gather(*future_map.values())
            # Map the results back to their respective keys
            for key, result in zip(future_map.keys(), results):
                all_results[key] = result
        return all_results

    def generate_function_call(self, model: str, user_msg, candidate_functions: List[Dict],
                               system_msg: Optional[str] = None) -> FunctionCall:

        output = self._chat_completion(model, user_msg, system_msg, candidate_functions)
        return self._parse_function_call(output)

    async def a_generate_function_call(self, model: str, user_msg, candidate_functions: List[Dict],
                                       system_msg: Optional[str] = None) -> FunctionCall:
        output = await self._a_chat_completion(model, user_msg, system_msg, candidate_functions)
        return self._parse_function_call(output)

    def _parse_function_call(self, output: Dict) -> FunctionCall:
        function_call = output.get("choices")[0]["message"]["function_call"]
        if function_call:
            function_call["arguments"] = json.loads(function_call["arguments"])
            return FunctionCall(**function_call)


    async def parse_text_to_cls(content: str,output_data_mapping: dict,) -> Any:
        instruct_content = ResultParser.parse_data_with_mapping(content, output_data_mapping)
        return instruct_content
