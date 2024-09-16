import unittest
from unittest.mock import MagicMock
import numpy as np

from thinker_ai.agent.document_store.vector_database import FAISSVectorDatabase, InMemoryVectorDatabase


class TestFAISSVectorDatabase(unittest.TestCase):

    def setUp(self):
        # 初始化 FAISSVectorDatabase 对象
        self.db = FAISSVectorDatabase(dimension=1536)

    def test_insert(self):
        # 插入数据
        self.db.insert("Test text")
        self.assertEqual(len(self.db.metadata), 1)
        self.assertEqual(len(self.db.vector_id_list), 1)

    def test_search(self):
        # 插入数据
        self.db.insert("Test text")
        # 搜索结果
        results = self.db.search("Query text", top_k=1)
        self.assertEqual(len(results), 1)

    def test_delete(self):
        # 插入数据
        self.db.insert("Test text")
        vector_id = self.db.vector_id_list[0]

        # 删除向量
        self.db.delete(vector_id)
        self.assertFalse(self.db.metadata[vector_id]['is_valid'])

    def test_update(self):
        # 插入数据
        self.db.insert("Test text")
        vector_id = self.db.vector_id_list[0]

        self.db.update(vector_id, "Updated text")
        results = self.db.search("Test text", top_k=1)
        result = results[0][0]
        self.assertEqual(result, "Updated text")

    def test_clear(self):
        # 插入数据
        self.db.insert("Test text")
        self.db.clear()

        # 检查数据库是否清空
        self.assertEqual(len(self.db.metadata), 0)
        self.assertEqual(len(self.db.vector_id_list), 0)


class TestInMemoryVectorDatabase(unittest.TestCase):

    def setUp(self):
        # 初始化 InMemoryVectorDatabase 对象
        self.db = InMemoryVectorDatabase()

    def test_insert(self):
        # 插入数据
        self.db.insert("Test text")
        self.assertEqual(len(self.db.vectors), 1)

    def test_insert_batch(self):
        self.db.clear()
        # 插入批量数据
        self.db.insert_batch(["Text 1", "Text 2", "Text 3"])
        self.assertEqual(len(self.db.vectors), 3)

    def test_search(self):
        # 插入数据
        self.db.insert("Test text")

        # 模拟搜索结果
        results = self.db.search("Query text", top_k=1)
        result = results[0][0]
        self.assertEqual(result, "Test text")

    def test_delete(self):
        # 插入数据
        self.db.insert("Test text")
        vector_id = list(self.db.vectors.keys())[0]

        # 删除向量
        self.db.delete(vector_id)
        self.assertNotIn(vector_id, self.db.vectors)

    def test_update(self):
        # 插入数据
        self.db.insert("Test text")
        vector_id = list(self.db.vectors.keys())[0]

        # 更新向量
        self.db.update(vector_id, "Updated text")
        results = self.db.search("Test text", top_k=1)
        result = results[0][0]
        self.assertEqual(result, "Updated text")

    def test_clear(self):
        # 插入数据
        self.db.insert("Test text")
        self.db.clear()

        # 检查数据库是否清空
        self.assertEqual(len(self.db.vectors), 0)


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()