from typing import Optional

from thinker_ai.actions.action_result import ActionResult
from thinker_ai.llm.llm_factory import get_llm
from thinker_ai.utils.action_result_parser import ActionResultParser


# @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _a_generate_action_output(user_msg: str,
                                    output_data_mapping: dict,
                                    system_msg: Optional[str] = None) -> ActionResult:
    content = await get_llm().a_generate(user_msg, system_msg, stream=True)
    instruct_content = ActionResultParser.parse_data_with_mapping(content, output_data_mapping)
    return ActionResult(content, instruct_content)
