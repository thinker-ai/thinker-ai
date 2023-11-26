import json
from dataclasses import dataclass

from typing import Type, Dict, Any, Tuple

from thinker_ai.actions import ActionResult
from thinker_ai.actions.action import BaseAction


@dataclass
class ActionMessage:
    action_result: ActionResult
    role: str = 'user'  # system / user / assistant
    cause_by: Type[BaseAction] = None  # 假设 BaseAction 是一个有效的类型
    sent_from: str = ""
    send_to: str = ""

    def serialize(self) -> str:
        result = {
            "action_result": self.action_result.content,
            "role": self.role,
            "cause_by": self.cause_by,
            "sent_from": self.sent_from,
            "send_to": self.send_to
        }
        return json.dumps(result,ensure_ascii=False)

    # 自定义反序列化方法
    @classmethod
    def deserialize(cls, json_str: str, serialize_mapping: Dict[str, Tuple[Type[Any], Any]]) -> "ActionMessage":
        data = json.loads(json_str)
        action_result = ActionResult.loads(data['action_result'], serialize_mapping)
        return cls(action_result=action_result, role=data['role'], cause_by=data['cause_by'],
                   sent_from=data['sent_from'], send_to=data['send_to'])
