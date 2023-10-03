from typing import Optional

from thinker_ai.action.action_output import ActionOutput

from thinker_ai.llm.llm_factory import get_llm


# @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _a_generate_action_output(user_msg: str, output_class_name: str,
                                    output_data_mapping: dict,
                                    system_msg: Optional[str] = None) -> ActionOutput:
    content = await get_llm().a_generate(user_msg, system_msg, stream=True)
    instruct_content = ActionOutput.parse_data_with_class(content, output_class_name, output_data_mapping)
    return ActionOutput(content, instruct_content)