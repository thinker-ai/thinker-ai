import asyncio
import json
import time
from typing import NamedTuple, Optional, List, Dict

import openai
from openai.error import APIConnectionError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, after_log, wait_fixed, retry_if_exception_type

from thinker_ai.llm.llm_api import LLM_API, FunctionCall
from thinker_ai.llm.schema import PromptMessage
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

    def split_batches(self, batch: list[PromptMessage]) -> list[list[PromptMessage]]:
        return [batch[i: i + self.rpm] for i in range(0, len(batch), self.rpm)]

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

    def __init__(self, max_budget):
        self.max_budget = max_budget
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def update_cost(self, prompt_tokens, completion_tokens, model):
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


def log_and_reraise(retry_state):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    raise retry_state.outcome.exception()


class GPT_Config(BaseModel):
    type: str = None
    version: str = None
    api_key: str
    model: str = "gpt-4-0613"
    temperature: int = 0
    max_budget: float = 3.0
    max_tokens_rsp: int = 2048
    api_base: str = "https://api.openai.com/v1"
    rpm: int = 10


class GPT(LLM_API, metaclass=Singleton):
    """
    Check https://platform.openai.com/examples for examples
    """

    def __init__(self, config: GPT_Config):
        self.__init_openai(config)
        self.llm = openai
        self.model = config.model
        self.auto_max_tokens = False
        self._cost_manager = CostManager(config.max_budget)
        self.max_tokens_rsp = config.max_tokens_rsp
        self.temperature = config.temperature
        self.rateLimiter = RateLimiter(rpm=self.rpm)

    def __init_openai(self, config: GPT_Config):
        openai.api_key = config.api_key
        openai.api_base = config.api_base
        if config.type:
            openai.api_type = config.type
            openai.api_version = config.version
        self.rpm = config.rpm

    default_system_msg = 'You are a helpful assistant.'

    def generate(self, user_msg: str, system_msg: Optional[str] = None) -> str:
        output = self._chat_completion(user_msg, system_msg)
        return self._get_choice_text(output)

    async def a_generate(self, user_msg: str, system_msg: Optional[str] = None, stream=False) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._a_chat_completion_stream(user_msg, system_msg)
        rsp = await self._a_chat_completion(user_msg, system_msg)
        return self._get_choice_text(rsp)

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["role"] == "assistant"])

    def _get_choice_text(self, rsp: dict) -> str:
        return rsp.get("choices")[0]["message"]["content"]

    async def _a_chat_completion_stream(self, user_msg: str, system_msg: Optional[str] = None) -> str:
        prompt = PromptMessage(user_msg, system_msg)
        response = await openai.ChatCompletion.acreate(**self._cons_kwargs(prompt.to_dicts()), stream=True)

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
        usage = self._calc_usage(prompt.to_dicts(), full_reply_content)
        self._update_costs(usage)
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], candidate_functions: Optional[List[Dict]] = None) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.get_max_tokens(messages),
            "n": 1,
            "stop": None,
            "temperature": self.temperature,
        }
        if candidate_functions:
            kwargs["functions"] = candidate_functions
        kwargs["timeout"] = 3
        return kwargs

    def _chat_completion(self, user_msg: str, system_msg: Optional[str] = None,
                         candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        rsp = self.llm.ChatCompletion.create(**self._cons_kwargs(prompt.to_dicts(), candidate_functions))
        self._update_costs(rsp.get("usage"))
        return rsp

    async def _a_chat_completion(self, user_msg: str, system_msg: Optional[str] = None,
                                 candidate_functions: Optional[List[Dict]] = None) -> dict:
        prompt = PromptMessage(user_msg, system_msg)
        rsp = await self.llm.ChatCompletion.acreate(**self._cons_kwargs(prompt.to_dicts(), candidate_functions))
        self._update_costs(rsp.get("usage"))
        return rsp

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        after=after_log(logger, logger.level('WARNING').name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    def _calc_usage(self, messages: list[dict], rsp: str) -> dict:
        usage = {}
        try:
            prompt_tokens = count_message_tokens(messages, self.model)
            completion_tokens = count_string_tokens(rsp, self.model)
            usage['prompt_tokens'] = prompt_tokens
            usage['completion_tokens'] = completion_tokens
            return usage
        except Exception as e:
            logger.error("usage calculation failed!", e)

    async def a_generate_batch(self, user_msgs: list[str],sys_msg:Optional[str] = None) -> list[str]:
        """仅返回纯文本"""
        messages: list[dict] = await self._a_completion_batch(user_msgs, sys_msg)
        results = []
        for message in messages:
            result = self._get_choice_text(message)
            results.append(result)
            logger.info(f"Result of task: {result}")
        return results

    async def _a_completion_batch(self, user_msgs: list[str], sys_msg: Optional[str] = None) -> list[dict]:
        batch_prompt = [PromptMessage(user_msg, sys_msg) for user_msg in user_msgs]
        split_batches: list[list[PromptMessage]] = self.rateLimiter.split_batches(batch_prompt)
        all_results_dict: dict[int, dict] = {}  # 使用字典暂存结果
        index_offset = 0  # 用于跟踪处理到哪个小批次

        for small_batch in split_batches:
            await self.rateLimiter.wait_if_needed(len(small_batch))
            # 使用 enumerate 记录每个请求的索引
            future = [self._a_chat_completion(prompt.user_message, prompt.system_message) for prompt in small_batch]
            indexes = [i + index_offset for i, _ in enumerate(small_batch)]
            # 为了保留原始顺序，分别等待每个future的结果
            for i, fut in zip(indexes, future):
                result = await fut
                logger.info(result)
                all_results_dict[i] = result
            index_offset += len(small_batch)  # 更新索引偏移量以处理下一个小批次

        # 将字典转换为列表
        all_results = [all_results_dict[i] for i in range(len(user_msgs))]

        return all_results

    def generate_function_call(self, user_msg, candidate_functions: List[Dict],
                               system_msg: Optional[str] = None) -> FunctionCall:

        output = self._chat_completion(user_msg, system_msg, candidate_functions)
        return self._parse_function_call(output)

    async def a_generate_function_call(self, user_msg, candidate_functions: List[Dict],
                                       system_msg: Optional[str] = None) -> FunctionCall:
        output = await self._a_chat_completion(user_msg, system_msg, candidate_functions)
        return self._parse_function_call(output)

    def _parse_function_call(self, output: Dict) -> FunctionCall:
        function_call = output.get("choices")[0]["message"]["function_call"]
        if function_call:
            function_call["arguments"] = json.loads(function_call["arguments"])
            return FunctionCall(**function_call)

    def _update_costs(self, usage: dict):
        try:
            prompt_tokens = int(usage['prompt_tokens'])
            completion_tokens = int(usage['completion_tokens'])
            self._cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
        except Exception as e:
            logger.error("updating costs failed!", e)

    def get_costs(self) -> Costs:
        return self._cost_manager.get_costs()

    def get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.max_tokens_rsp
        return get_max_completion_tokens(messages, self.model, self.max_tokens_rsp)
