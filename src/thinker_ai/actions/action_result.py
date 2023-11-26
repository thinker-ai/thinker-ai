from typing import Any, Dict, Type, Tuple

from thinker_ai.utils.action_result_parser import ActionResultParser


class ActionResult:
    def __init__(self, content: str,instruct_content: Any):
        self.content = content
        self.instruct_content = instruct_content

    @classmethod
    def loads(cls, content: str, serialize_mapping: Dict[str, Tuple[Type[Any], Any]]) -> "ActionResult":
        parsed_data = ActionResultParser.parse_data_with_mapping(content, serialize_mapping)
        return cls(content, parsed_data)
