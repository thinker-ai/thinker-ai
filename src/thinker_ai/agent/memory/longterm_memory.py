import json

from pathlib import Path
from typing import Iterable, Type, Dict, Any, Tuple

from thinker_ai.actions import BaseAction
from thinker_ai.work_flow.tasks import TaskMessage
from thinker_ai.agent.memory.memory import Memory

ACTION_SUBCLASSES = {cls.__name__: cls for cls in BaseAction.__subclasses__()}


class LongTermMemory(Memory):
    def __init__(self, file_path: str, serialize_mapping: Dict[str, Tuple[Type[Any], Any]]):
        super().__init__()
        self._file_path = file_path
        self._load_data_from_file(serialize_mapping)

    def _save_data_to_file(self):
        data = {
            "index": {
                key.__name__: [msg.serialize() for msg in value]
                for key, value in self.index.items()
            },
            "storage": [msg.serialize() for msg in self.storage]
        }
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, 'w') as file:
            json.dump(obj=data, fp=file, ensure_ascii=False)

    def _load_data_from_file(self, serialize_mapping: Dict[str, Tuple[Type[Any], Any]]):
        try:
            with open(self._file_path, 'r') as file:
                data = json.load(file)

            self.index.update({
                ACTION_SUBCLASSES[key]: [TaskMessage.deserialize(msg_data, serialize_mapping) for msg_data in value]
                for key, value in data.get("index", {}).items()
            })

            self.storage = [TaskMessage.deserialize(msg_data, serialize_mapping) for msg_data in
                            data.get("storage", [])]
        except IOError:
            pass

    def add(self, message: TaskMessage):
        """Add a new message to storage, while updating the index"""
        super().add(message)
        self._save_data_to_file()

    def add_batch(self, messages: Iterable[TaskMessage]):
        for message in messages:
            super().add(message)
        self._save_data_to_file()

    def delete(self, message: TaskMessage):
        """Delete the specified message from storage, while updating the index"""
        super().delete(message)
        self._save_data_to_file()

    def clear(self):
        """Clear storage and index"""
        super().clear()
        self._save_data_to_file()
