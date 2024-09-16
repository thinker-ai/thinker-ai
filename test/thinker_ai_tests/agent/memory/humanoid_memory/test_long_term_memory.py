import unittest

from thinker_ai.agent.memory.humanoid_memory.graph_DNC import GraphDNC
from thinker_ai.agent.memory.humanoid_memory.long_term_memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.dnc = GraphDNC()
        self.ltm = LongTermMemory(dnc=self.dnc)

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
        ltm_new = LongTermMemory(dnc=self.dnc)
        ltm_new.load()
        self.assertEqual(len(ltm_new.knowledge_base), 1)
        self.assertEqual(ltm_new.retrieve_knowledge("AI"), "Artificial Intelligence")


if __name__ == '__main__':
    unittest.main()
