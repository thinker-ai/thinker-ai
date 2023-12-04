import asyncio
import json
import time
from typing import NamedTuple, Optional, List, Dict, Union

from openai import OpenAI, AsyncOpenAI, APIConnectionError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, after_log, wait_fixed, retry_if_exception_type

from thinker_ai.llm.function_call import FunctionCall
from thinker_ai.llm.gpt_schema import PromptMessage
from thinker_ai.utils.logs import logger
from thinker_ai.utils.singleton import Singleton
from thinker_ai.utils.token_counter import (
    TOKEN_COSTS,
    count_message_tokens,
    count_string_tokens,
    get_max_completion_tokens,
)


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


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(metaclass=Singleton):
    """计算使用接口的开销"""

    def __init__(self, max_budget, auto_max_tokens: bool = False, max_tokens_rsp: int = 2048):
        self.max_budget = max_budget
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.auto_max_tokens = auto_max_tokens
        self.max_tokens_rsp = max_tokens_rsp

    def update_cost(self, model: str, prompt_tokens, completion_tokens):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        cost = (prompt_tokens * TOKEN_COSTS[model]["prompt"] + completion_tokens * TOKEN_COSTS[model][
            "completion"]) / 1000
        self.total_cost += cost
        logger.info(
            f"Total running cost: ${self.total_cost:.3f} | Max budget: ${self.max_budget:.3f} | "
            f"Current cost: ${cost:.3f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_costs(self) -> Costs:
        """获得所有开销"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)

    def update_costs(self, model: str, usage: dict):
        try:
            prompt_tokens = int(usage['prompt_tokens'])
            completion_tokens = int(usage['completion_tokens'])
            self.update_cost(model, prompt_tokens, completion_tokens)
        except Exception as e:
            logger.error("updating costs failed!", e)

    def get_max_tokens(self, model: str, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.max_tokens_rsp
        return get_max_completion_tokens(messages, model, self.max_tokens_rsp)

    def calc_usage(self, model: str, messages: list[dict], rsp: str) -> dict:
        usage = {}
        try:
            prompt_tokens = count_message_tokens(messages, model)
            completion_tokens = count_string_tokens(rsp, model)
            usage['prompt_tokens'] = prompt_tokens
            usage['completion_tokens'] = completion_tokens
            return usage
        except Exception as e:
            logger.error("usage calculation failed!", e)


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

    def __init__openai(self, config, openai: Union[OpenAI, AsyncOpenAI]):
        openai.api_base = config.api_base
        openai.proxy = config.proxy
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
        response = await self.a_llm.chat.completions.create(model=model,
                                                            **self._cons_kwargs(model, prompt.to_dicts()), stream=True)
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        async for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            if "content" in chunk_message:
                print(chunk_message["content"], end="")
        print()

        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        usage = self._cost_manager.calc_usage(model, prompt.to_dicts(), full_reply_content)
        self._cost_manager.update_costs(model, usage)
        return full_reply_content

    def _cons_kwargs(self, model: str, messages: list[dict], candidate_functions: Optional[List[Dict]] = None) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._cost_manager.get_max_tokens(model, messages),
            "n": 1,
            "stop": None,
            "temperature": self.temperature,
        }
        if candidate_functions:
            kwargs["functions"] = candidate_functions
        kwargs["timeout"] = 3
        return kwargs

    def _chat_completion(self, model: str, user_msg: str, system_msg: Optional[str] = None,
                         candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        rsp = self.llm.chat.completions.create(model=model,
                                               **self._cons_kwargs(model, prompt.to_dicts(), candidate_functions))
        self._cost_manager.update_costs(model, rsp.usage.dict())
        return rsp.dict()

    async def _a_chat_completion(self, model: str, user_msg: str, system_msg: Optional[str] = None,
                                 candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        rsp = await self.a_llm.chat.completions.create(model=model,
                                                       **self._cons_kwargs(model, prompt.to_dicts(),
                                                                           candidate_functions))
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
