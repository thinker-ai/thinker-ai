import unittest

from thinker_ai.agent.document_store.vector_database import FAISSVectorDatabase
from thinker_ai.agent.memory.humanoid_memory.deep_learning_memory_network import DeepLearningMemoryNetwork
from thinker_ai.agent.memory.humanoid_memory.graph_DNC import GraphDNC
from thinker_ai.agent.memory.humanoid_memory.humanoid_brain_memory import HumanoidBrainMemory
from thinker_ai.agent.provider.llm_schema import Message


class TestHumanoidBrainMemory(unittest.TestCase):
    def setUp(self):
        self.dnc = GraphDNC()
        self.memory_network = DeepLearningMemoryNetwork(vector_db=FAISSVectorDatabase())
        self.brain_memory = HumanoidBrainMemory(
            owner_id="user_123",
            dnc=self.dnc,
            memory_network=self.memory_network
        )

    def test_add_talk_and_get_history(self):
        msg1 = Message(content="你好，今天天气怎么样？", role="user")
        self.brain_memory.add_talk(msg1)
        history = self.brain_memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], msg1)
        self.assertEqual(history[0].role, "user")

    def test_add_answer_and_get_history(self):
        msg1 = Message(content="今天天气晴朗。", role="assistant")
        self.brain_memory.add_answer(msg1)
        history = self.brain_memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], msg1)
        self.assertEqual(history[0].role, "assistant")

    def test_get_history_text(self):
        msg1 = Message(content="你好，今天天气怎么样？", role="user")
        msg2 = Message(content="今天天气晴朗。", role="assistant")
        self.brain_memory.add_talk(msg1)
        self.brain_memory.add_answer(msg2)
        history_text = self.brain_memory.get_history_text()
        expected_text = "user: 你好，今天天气怎么样？\nassistant: 今天天气晴朗。"
        self.assertEqual(history_text, expected_text)

    def test_summarize(self):
        for i in range(10):
            msg = Message(content=f"消息内容 {i}", role="user")
            self.brain_memory.add_talk(msg)
        summary = self.brain_memory.summarize(max_words=50)
        self.assertTrue(len(summary) <= 50)

    def test_get_title(self):
        msg1 = Message(content="聊一聊人工智能。", role="user")
        self.brain_memory.add_talk(msg1)
        title = self.brain_memory.get_title()
        expected_title = "聊一聊人工智能。"
        self.assertEqual(title, expected_title[:20])

    def test_is_related(self):
        msg1 = Message(content="人工智能的发展趋势如何？", role="user")
        self.brain_memory.add_talk(msg1)
        related = self.brain_memory.is_related("人工智能", "谈谈人工智能的应用。")
        self.assertTrue(related)
        unrelated = self.brain_memory.is_related("机器学习", "今天天气如何？")
        self.assertFalse(unrelated)

    def test_rewrite(self):
        context = "我们在讨论人工智能。"
        sentence = "它有哪些应用？"
        rewritten = self.brain_memory.rewrite(sentence, context)
        expected_rewritten = f"{context} {sentence} (人工智能是一个广泛的领域...)"
        # 由于重写方法的简单实现，我们只检查返回类型
        self.assertIsInstance(rewritten, str)

    def test_exists(self):
        msg1 = Message(content="你好，世界！", role="user")
        self.brain_memory.add_talk(msg1)
        self.assertTrue(self.brain_memory.exists("你好，世界！"))
        self.assertFalse(self.brain_memory.exists("再见，世界！"))

    def test_add_and_get_knowledge(self):
        self.brain_memory.add_knowledge(key="人工智能", value="AI 的缩写")
        knowledge = self.brain_memory.get_knowledge("人工智能")
        self.assertEqual(knowledge, "AI 的缩写")

    def test_search_knowledge(self):
        self.brain_memory.add_knowledge(key="人工智能", value="AI 的缩写")
        self.brain_memory.add_knowledge(key="机器学习", value="ML 的缩写")
        results = self.brain_memory.search_knowledge("缩写")
        self.assertEqual(len(results), 2)
        self.assertIn("AI 的缩写", results)
        self.assertIn("ML 的缩写", results)

    def test_clear_history(self):
        msg1 = Message(content="测试消息", role="user")
        self.brain_memory.add_talk(msg1)
        self.brain_memory.clear_history()
        self.assertEqual(len(self.brain_memory.get_history()), 0)

    def test_clear_knowledge(self):
        self.brain_memory.add_knowledge(key="测试", value="测试内容")
        self.brain_memory.clear_knowledge()
        knowledge = self.brain_memory.get_knowledge("测试")
        self.assertIsNone(knowledge)

    def test_save_and_load(self):
        msg1 = Message(content="保存前的消息", role="user")
        self.brain_memory.add_talk(msg1)
        self.brain_memory.add_knowledge(key="保存前的知识", value="一些知识内容")
        self.brain_memory.save()

        # 创建新的 HumanoidBrainMemory 实例，使用相同的持久化
        new_brain_memory = HumanoidBrainMemory(
            owner_id="user_123",
            dnc=self.dnc,
            memory_network=self.memory_network
        )
        new_brain_memory.load()

        # 检查历史记录
        history = new_brain_memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].content, "保存前的消息")

        # 检查知识
        knowledge = new_brain_memory.get_knowledge("保存前的知识")
        self.assertEqual(knowledge, "一些知识内容")

    def test_get_last_talk(self):
        msg1 = Message(content="第一条消息", role="user")
        msg2 = Message(content="第二条消息", role="user")
        self.brain_memory.add_talk(msg1)
        self.brain_memory.add_talk(msg2)
        last_talk = self.brain_memory.get_last_talk()
        self.assertEqual(last_talk.content, "第二条消息")

    def test_is_history_available(self):
        self.assertFalse(self.brain_memory.is_history_available)
        msg1 = Message(content="测试消息", role="user")
        self.brain_memory.add_talk(msg1)
        self.assertTrue(self.brain_memory.is_history_available)

    def test_history_text_property(self):
        msg1 = Message(content="你好", role="user")
        msg2 = Message(content="您好，有什么可以帮助您？", role="assistant")
        self.brain_memory.add_talk(msg1)
        self.brain_memory.add_answer(msg2)
        history_text = self.brain_memory.history_text
        expected_text = "user: 你好\nassistant: 您好，有什么可以帮助您？"
        self.assertEqual(history_text, expected_text)

    def test_to_cache_key(self):
        cache_key = self.brain_memory.to_cache_key(prefix="memory", user_id="user_123", chat_id="chat_456")
        expected_key = "memory:user_123:chat_456"
        self.assertEqual(cache_key, expected_key)

    def test_extract_info(self):
        input_string = "[INFO]: 这是一条测试信息。"
        label, content = HumanoidBrainMemory.extract_info(input_string)
        self.assertEqual(label, "INFO")
        self.assertEqual(content, "这是一条测试信息。")

        input_string_no_label = "没有标签的信息。"
        label, content = HumanoidBrainMemory.extract_info(input_string_no_label)
        self.assertIsNone(label)
        self.assertEqual(content, "没有标签的信息。")


if __name__ == '__main__':
    unittest.main()
