import json

from pathlib import Path
from typing import Iterable

from thinker_ai.actions.action import BaseAction
from thinker_ai.actions.action_output import ActionOutput
from thinker_ai.llm.schema import Message
from thinker_ai.memory.memory import Memory


ACTION_SUBCLASSES = {cls.__name__: cls for cls in BaseAction.__subclasses__()}

def _serialize_message(msg: Message) -> dict:
    class_name = msg.instruct_content.__class__.__name__
    msg_dict = {
        "content": msg.content,
        "instruct_content_class": class_name if class_name != "NoneType" else None,
        "role": msg.role,
        "cause_by": msg.cause_by.__name__ if msg.cause_by else None,
        "sent_from": msg.sent_from,
        "send_to": msg.send_to
    }
    return msg_dict


def _deserialize_message(msg_data: dict, data_mappings: dict) -> Message:
    """Converts a dictionary representation back to a Message instance."""
    class_name = msg_data["instruct_content_class"]
    return Message(
        content=msg_data["content"],
        instruct_content=ActionOutput.parse_data_with_class(msg_data["content"], class_name, data_mappings.get(class_name))
        if class_name and data_mappings.get(class_name) else None,
        role=msg_data["role"],
        cause_by=ACTION_SUBCLASSES.get(msg_data["cause_by"]),
        sent_from=msg_data["sent_from"],
        send_to=msg_data["send_to"]
    )


class LongTermMemory(Memory):
    def __init__(self, file_path: str,data_mappings:dict):
        super().__init__()
        self._file_path = file_path
        self._load_data_from_file(data_mappings)

    def _save_data_to_file(self):
        data = {
            "index": {
                key.__name__: [_serialize_message(msg) for msg in value]
                for key, value in self.index.items()
            },
            "storage": [_serialize_message(msg) for msg in self.storage]
        }
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, 'w') as file:
            json.dump(obj=data, fp=file, ensure_ascii=False)

    def _load_data_from_file(self,data_mappings:dict):
        try:
            with open(self._file_path, 'r') as file:
                data = json.load(file)

            self.index.update({
                ACTION_SUBCLASSES[key]: [_deserialize_message(msg_data,data_mappings) for msg_data in value]
                for key, value in data.get("index", {}).items()
            })

            self.storage = [_deserialize_message(msg_data,data_mappings) for msg_data in data.get("storage", [])]
        except IOError:
            pass

    def add(self, message: Message):
        """Add a new message to storage, while updating the index"""
        super().add(message)
        self._save_data_to_file()

    def add_batch(self, messages: Iterable[Message]):
        for message in messages:
            super().add(message)
        self._save_data_to_file()

    def delete(self, message: Message):
        """Delete the specified message from storage, while updating the index"""
        super().delete(message)
        self._save_data_to_file()

    def clear(self):
        """Clear storage and index"""
        super().clear()
        self._save_data_to_file()
