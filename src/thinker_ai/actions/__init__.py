from typing import Optional, Any

from thinker_ai.llm import gpt
from thinker_ai.actions.result_parser import ResultParser


# @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _a_generate_action_output(user_msg: str,
                                    output_data_mapping: dict,
                                    system_msg: Optional[str] = None) -> Any:
    content = await gpt.a_generate(user_msg, system_msg, stream=True)
    instruct_content = ResultParser.parse_data_with_mapping(content, output_data_mapping)
    return instruct_content
