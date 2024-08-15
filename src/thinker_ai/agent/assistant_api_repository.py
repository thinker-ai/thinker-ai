import json
from abc import ABC
from json import JSONDecodeError
from typing import Optional

from thinker_ai.configs.const import PROJECT_ROOT
from threading import Lock


class AssistantRepository:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT / 'data/assistants.json') -> "AssistantRepository":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(filepath)
        return cls._instance

    def __init__(self, filepath: str):
        if not AssistantRepository._instance:
            self.filepath = filepath
            self._data = self._load_data()
            AssistantRepository._instance = self
        else:
            raise Exception("Attempting to instantiate a singleton class.")

    def _load_data(self):
        try:
            with open(self.filepath, 'r') as file:
                return json.load(file)
        except (FileNotFoundError,JSONDecodeError):
            return {}

    def _save_data(self):
        with open(self.filepath, 'w') as file:
            json.dump(self._data, file, indent=4)

    def add(self,name: str,assistant_id:str):
        self._data[name] = assistant_id
        self._save_data()

    def update(self,name: str,assistant_id:str):
        if name in self._data.keys():
            self._data[name] = assistant_id
            self._save_data()

    def delete(self, name):
        self._data = {k: v for k, v in self._data.items() if k != name}
        self._save_data()

    def get_by_name(self, name: str) -> Optional[str]:
        return self._data.get(name)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
