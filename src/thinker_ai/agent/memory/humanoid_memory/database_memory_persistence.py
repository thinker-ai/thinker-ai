# 假设使用 SQLite 数据库
import sqlite3
import pickle
from typing import Any

from thinker_ai.agent.memory.humanoid_memory.persistence import MemoryPersistence


class DatabaseMemoryPersistence(MemoryPersistence):
    """
    使用数据库进行持久化的实现。
    """

    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.connection = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.connection.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY,
                data BLOB
            )
        ''')
        self.connection.commit()

    def save(self, data: Any):
        cursor = self.connection.cursor()
        serialized_data = pickle.dumps(data)
        cursor.execute(f'DELETE FROM {self.table_name}')
        cursor.execute(f'INSERT INTO {self.table_name} (data) VALUES (?)', (serialized_data,))
        self.connection.commit()

    def load(self) -> Any:
        cursor = self.connection.cursor()
        cursor.execute(f'SELECT data FROM {self.table_name} ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        else:
            return None