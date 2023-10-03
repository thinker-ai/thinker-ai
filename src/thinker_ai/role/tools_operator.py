import re
from typing import Union, Optional, Any, Dict, List
from pydantic import BaseModel

from thinker_ai.context import Context
from thinker_ai.llm.llm_factory import get_llm
from thinker_ai.role.role import Role, RoleConfig
from thinker_ai.tools.tools_register import ToolsRegister


class LLM_Response(BaseModel):
    content: str


def _execute_generated_code(generated_code: str) -> Any:
    try:
        locals_dict = {}
        exec(generated_code, globals(), locals_dict)
        if 'output_data' in locals_dict:
            return locals_dict['output_data']
    except Exception as e:
        # Print the exception message for debugging purposes
        print("Exception occurred:", e)
        raise e


class AgentWithTools(Role):
    tools_register = ToolsRegister()

    def __init__(self, context: Context, role_config: RoleConfig):
        super().__init__(role_config)
        self.context = context

    def _extract_generated_code(self, llm_response: LLM_Response) -> Optional[str]:
        code_pattern = re.compile(r"##code:\n'''([\s\S]*?)'''")
        code_texts = code_pattern.findall(llm_response.content)
        if not code_texts:
            code_pattern = re.compile(r"##code:\n'''python([\s\S]*?)'''")
            code_texts = code_pattern.findall(llm_response.content)
        if not code_texts:
            return None
        return code_texts[0]

    def _extract_generated_think(self, llm_response: LLM_Response):
        think_pattern = re.compile(r"##think:\n'''([\s\S]*?)'''")
        think_texts = think_pattern.findall(llm_response.content)
        if not think_texts:
            raise "think_text not found"
        return think_texts[0]
