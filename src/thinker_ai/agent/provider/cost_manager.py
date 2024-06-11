import re
from typing import NamedTuple, Any, List, Union

from openai.types import CompletionUsage
from pydantic import BaseModel

from thinker_ai.common.logs import logger

from thinker_ai.agent.provider.token_counter import (
    TOKEN_COSTS,
    FIREWORKS_GRADE_TOKEN_COSTS,
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)
from thinker_ai.configs.config import config


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(BaseModel):
    token_costs: dict[str, dict[str, float]] = TOKEN_COSTS  # different model's token cost
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: int = 0
    total_budget: int = 0

    def update_costs(self, model: str, usage: Union[dict, BaseModel]):
        """update each request's token cost
        Args:
            model (str): model name or in some scenarios called endpoint
            local_calc_usage (bool): some models don't calculate usage, it will overwrite LLMConfig.calc_usage
        """
        usage = usage.model_dump() if isinstance(usage, BaseModel) else usage
        if usage:
            try:
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                self._update_cost(prompt_tokens, completion_tokens, model)
            except Exception as e:
                logger.error(f"{self.__class__.__name__} updates costs failed! exp: {e}")

    def _update_cost(self, prompt_tokens, completion_tokens, model):
        """
            Update the total cost, prompt tokens, and completion tokens.

            Args:
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens used in the completion.
            model (str): The model used for the API call.
            """
        if prompt_tokens + completion_tokens == 0 or not model:
            return
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        if model not in self.token_costs:
            logger.warning(f"Model {model} not found in TOKEN_COSTS.")
            return

        cost = (
                       prompt_tokens * self.token_costs[model]["prompt"]
                       + completion_tokens * self.token_costs[model]["completion"]
               ) / 1000
        self.total_cost += cost
        logger.info(
            f"Total running cost: ${self.total_cost:.3f} | Max budget: ${config.llm.max_budget:.3f} | "
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
        """Get all costs"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)

    def calc_usage(self, model: str, messages: list[dict], rsp: str) -> CompletionUsage:
        try:
            prompt_tokens = count_input_tokens(messages, model)
            completion_tokens = count_output_tokens(rsp, model)
            total_tokens = prompt_tokens + completion_tokens
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens)
            return usage
        except Exception as e:
            logger.warning(f"usage calculation failed: {e}")

    def get_max_tokens(self, model: str, messages: list[dict]):
        if not config.llm.auto_max_tokens:
            return config.llm.max_token
        # FIXME
        # https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, model, config.llm.max_tokens_rsp), 4096)


class TokenCostManager(CostManager):
    """open llm model is self-host, it's free and without cost"""

    def _update_cost(self, prompt_tokens, completion_tokens, model):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")


class FireworksCostManager(CostManager):
    def model_grade_token_costs(self, model: str) -> dict[str, float]:
        def _get_model_size(model: str) -> float:
            size = re.findall(".*-([0-9.]+)b", model)
            size = float(size[0]) if len(size) > 0 else -1
            return size

        if "mixtral-8x7b" in model:
            token_costs = FIREWORKS_GRADE_TOKEN_COSTS["mixtral-8x7b"]
        else:
            model_size = _get_model_size(model)
            if 0 < model_size <= 16:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["16"]
            elif 16 < model_size <= 80:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["80"]
            else:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["-1"]
        return token_costs

    def _update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        token_costs = self.model_grade_token_costs(model)
        cost = (prompt_tokens * token_costs["prompt"] + completion_tokens * token_costs["completion"]) / 1000000
        self.total_cost += cost
        logger.info(
            f"Total running cost: ${self.total_cost:.4f}"
            f"Current cost: ${cost:.4f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )
