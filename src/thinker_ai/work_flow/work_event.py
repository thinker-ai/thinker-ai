import json
from dataclasses import dataclass

from typing import Type, Dict, Any, Tuple

from thinker_ai.actions import ActionResult


@dataclass
class WorkEvent:
    id:str
    source_id:str
    name:str
    payload: Any

    def serialize(self) -> str:
        result = {
            "id": self.id,
            "source_id": self.source_id,
            "name": self.name,
            "payload": self.payload.content
        }
        return json.dumps(result,ensure_ascii=False)

    @classmethod
    def deserialize(cls, json_str: str, serialize_mapping: Dict[str, Tuple[Type[Any], Any]]) -> "WorkEvent":
        event = json.loads(json_str)
        payload = ActionResult.loads(event['payload'], serialize_mapping)
        return cls(payload=payload, id=event['id'], name=event['name'],source_id=event['source_id'])
