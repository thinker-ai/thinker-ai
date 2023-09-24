from abc import ABC, abstractmethod
from typing import Optional, Tuple

from thinker_ai.action.action_output import ActionOutput
from thinker_ai.context import Context
from thinker_ai.llm.llm_factory import get_llm
from thinker_ai.utils.logs import logger


class Action(ABC):
    def __init__(self, context: Context):
        self.context = context
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
    @classmethod
    async def _a_generate_stream(self, user_msg: str, system_msg: Optional[str] = None) -> str:
        """Append default prefix"""
        content = await get_llm().a_generate_stream(user_msg, system_msg)
        logger.debug(content)
        return content
    @classmethod
    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def _a_generate_action_output(self, user_msg: str, output_class_name: str,
                                        output_data_mapping: dict,
                                        system_msg: Optional[str] = None) -> ActionOutput:
        content = await self._a_generate_stream(user_msg, system_msg)
        instruct_content = ActionOutput.parse_data_with_class(content, output_class_name, output_data_mapping)
        return ActionOutput(content, instruct_content)

    @abstractmethod
    async def run(self, *args, **kwargs):
        """The run method should be implemented in a subclass"""
