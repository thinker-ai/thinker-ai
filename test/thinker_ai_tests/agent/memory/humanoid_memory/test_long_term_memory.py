import unittest

from thinker_ai.agent.memory.humanoid_memory.long_term_memory import LongTermMemory
from thinker_ai.agent.memory.humanoid_memory.persistence import MemoryPersistence


class InMemoryPersistence(MemoryPersistence):
    """
    内存中的持久化实现，用于测试，避免实际的文件或数据库操作。
    """

    def __init__(self):
        self.data = None

    def save(self, data):
        self.data = data

    def load(self):
        return self.data


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.persistence = InMemoryPersistence()
        self.ltm = LongTermMemory(persistence=self.persistence)

    def test_store_and_retrieve_knowledge(self):
        key = "AI"
        value = "Artificial Intelligence"
        self.ltm.store_message(key, value)
        retrieved = self.ltm.retrieve_knowledge(key)
        self.assertEqual(retrieved, value)

    def test_search_knowledge(self):
        self.ltm.store_message("AI", "Artificial Intelligence")
        self.ltm.store_message("ML", "Machine Learning")
        results = self.ltm.search_knowledge("Intelligence")
        self.assertEqual(len(results), 1)
        self.assertIn("Artificial Intelligence", results)
        results = self.ltm.search_knowledge("Learning")
        self.assertEqual(len(results), 1)
        self.assertIn("Machine Learning", results)

    def test_is_related(self):
        self.ltm.store_message("AI", "Artificial Intelligence")
        self.assertTrue(self.ltm.is_related("AI", "Some text"))
        self.assertFalse(self.ltm.is_related("Quantum Computing", "Some text"))

    def test_clear(self):
        self.ltm.store_message("AI", "Artificial Intelligence")
        self.ltm.clear()
        self.assertEqual(len(self.ltm.knowledge_base), 0)

    def test_save_and_load(self):
        self.ltm.store_message("AI", "Artificial Intelligence")
        self.ltm.save()
        # 创建新的 LongTermMemory 实例，使用相同的持久化
        ltm_new = LongTermMemory(persistence=self.persistence)
        ltm_new.load()
        self.assertEqual(len(ltm_new.knowledge_base), 1)
        self.assertEqual(ltm_new.retrieve_knowledge("AI"), "Artificial Intelligence")


if __name__ == '__main__':
    unittest.main()
