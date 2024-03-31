
import re
from typing import NamedTuple, List, Dict

from thinker_ai.utils.logs import logger

from thinker_ai.agent.llm.token_counter import (
    TOKEN_COSTS,
    FIREWORKS_GRADE_TOKEN_COSTS,
    count_message_tokens,
    count_string_tokens,
    get_max_completion_tokens,
)

class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager:
    """计算使用接口的开销"""
    def __init__(self, max_budget, auto_max_tokens: bool = False, max_tokens_rsp: int = 2048):
        self.max_budget = max_budget
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.auto_max_tokens = auto_max_tokens
        self.max_tokens_rsp = max_tokens_rsp

    def update_cost(self,model: str, prompt_tokens, completion_tokens):
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
        if model not in TOKEN_COSTS:
            logger.warning(f"Model {model} not found in TOKEN_COSTS.")
            return

        cost = (
            prompt_tokens * TOKEN_COSTS[model]["prompt"]
            + completion_tokens * TOKEN_COSTS[model]["completion"]
        ) / 1000
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

    def get_max_tokens(self, model: str, messages: List[dict]):
        if not self.auto_max_tokens:
            return self.max_tokens_rsp
        return get_max_completion_tokens(messages, model, self.max_tokens_rsp)

    def calc_usage(self, model: str, messages: List[dict], rsp: str) -> dict:
        usage = {}
        try:
            prompt_tokens = count_message_tokens(messages, model)
            completion_tokens = count_string_tokens(rsp, model)
            usage['prompt_tokens'] = prompt_tokens
            usage['completion_tokens'] = completion_tokens
            return usage
        except Exception as e:
            logger.error("usage calculation failed!", e)


class TokenCostManager(CostManager):
    """open llm model is self-host, it's free and without cost"""

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
        logger.info(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")


class FireworksCostManager(CostManager):
    def model_grade_token_costs(self, model: str) -> Dict[str, float]:
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

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """
        Refs to `https://app.fireworks.ai/pricing` **Developer pricing**
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        token_costs = self.model_grade_token_costs(model)
        cost = (prompt_tokens * token_costs["prompt"] + completion_tokens * token_costs["completion"]) / 1000000
        self.total_cost += cost
        logger.info(
            f"Total running cost: ${self.total_cost:.4f}"
            f"Current cost: ${cost:.4f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )
