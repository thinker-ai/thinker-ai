import json
from typing import Any

from thinker_ai.agent.memory.humanoid_memory.persistence import MemoryPersistence


class FileMemoryPersistence(MemoryPersistence):
    """
    使用文件进行持久化的实现。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, data: Any):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self) -> Any:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None