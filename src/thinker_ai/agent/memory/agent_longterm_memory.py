import json
from pathlib import Path
from typing import Iterable, Dict, List
from thinker_ai.agent.memory.memory import Memory  # 假设这是经过修改的Memory类


class AgentLongTermMemory(Memory):
    def __init__(self, agent_id: str, file_path: str):
        super().__init__(agent_id)
        self._file_path = file_path
        self._load_data_from_file()

    def _save_data_to_file(self):
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, 'w', encoding='utf-8') as file:
            json.dump(obj=self.storage, fp=file, ensure_ascii=False, indent=4)

    def _load_data_from_file(self):
        try:
            with open(self._file_path, 'r', encoding='utf-8') as file:
                self.storage = json.load(file)
        except (IOError, json.JSONDecodeError):
            # 文件不存在或读取错误时，初始化storage为一个空字典
            self.storage = {}

    def add(self, topic: str, message: str):
        super().add(topic, message)
        self._save_data_to_file()

    def add_batch(self, topic: str, messages: Iterable[str]):
        super().add_batch(topic, messages)
        self._save_data_to_file()

    def get_by_keyword(self, topic: str, keyword: str) -> List[str]:
        return super().get_by_keyword(topic, keyword)

    def delete(self, topic: str, message: str):
        super().delete(topic, message)
        self._save_data_to_file()

    def clear(self):
        super().clear()
        self._save_data_to_file()
