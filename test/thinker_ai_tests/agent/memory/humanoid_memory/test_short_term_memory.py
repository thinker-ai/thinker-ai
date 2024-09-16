import unittest

from thinker_ai.agent.document_store.vector_database import FAISSVectorDatabase
from thinker_ai.agent.memory.humanoid_memory.deep_learning_memory_network import DeepLearningMemoryNetwork
from thinker_ai.agent.memory.humanoid_memory.short_term_memory import ShortTermMemory
from thinker_ai.agent.provider.llm_schema import Message


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.memory_network = DeepLearningMemoryNetwork(vector_db=FAISSVectorDatabase())
        self.stm = ShortTermMemory(memory_network=self.memory_network)

    def test_add_message(self):
        msg = Message(content="Hello", role="user")
        self.stm.add_message(msg)
        self.assertEqual(len(self.stm.messages), 1)
        self.assertEqual(self.stm.messages[0], msg)

    def test_get_history(self):
        msg1 = Message(content="Hello", role="user")
        msg2 = Message(content="Hi", role="assistant")
        self.stm.add_message(msg1)
        self.stm.add_message(msg2)
        history = self.stm.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], msg1)
        self.assertEqual(history[1], msg2)

    def test_get_history_text(self):
        msg1 = Message(content="Hello", role="user")
        msg2 = Message(content="Hi there!", role="assistant")
        self.stm.add_message(msg1)
        self.stm.add_message(msg2)
        history_text = self.stm.get_history_text()
        expected_text = "user: Hello\nassistant: Hi there!"
        self.assertEqual(history_text, expected_text)

    def test_get_recent_context(self):
        for i in range(10):
            msg = Message(content=f"Message {i}", role="user")
            self.stm.add_message(msg)
        recent_context = self.stm.get_recent_context(window_size=5)
        self.assertEqual(len(recent_context), 5)
        self.assertEqual(recent_context[0].content, "Message 5")
        self.assertEqual(recent_context[-1].content, "Message 9")

    def test_exists(self):
        msg = Message(content="Hello", role="user")
        self.stm.add_message(msg)
        self.assertTrue(self.stm.exists("Hello"))
        self.assertFalse(self.stm.exists("Hi"))

    def test_is_related(self):
        text1 = "Hello world"
        text2 = "Hi, how is the world today?"
        self.assertTrue(self.stm.is_related(text1, text2))
        text3 = "Goodbye"
        self.assertFalse(self.stm.is_related(text1, text3))

    def test_clear(self):
        msg = Message(content="Hello", role="user")
        self.stm.add_message(msg)
        self.stm.clear()
        self.assertEqual(len(self.stm.messages), 0)

    def test_save_and_load(self):
        msg1 = Message(content="Hello", role="user")
        msg2 = Message(content="Hi", role="assistant")
        self.stm.add_message(msg1)
        self.stm.add_message(msg2)
        self.stm.save()
        # 创建新的 ShortTermMemory 实例，使用相同的持久化
        stm_new = ShortTermMemory(memory_network=self.memory_network)
        stm_new.load()
        self.assertEqual(len(stm_new.messages), 2)
        self.assertEqual(stm_new.messages[0], msg1)
        self.assertEqual(stm_new.messages[1], msg2)


if __name__ == '__main__':
    unittest.main()
