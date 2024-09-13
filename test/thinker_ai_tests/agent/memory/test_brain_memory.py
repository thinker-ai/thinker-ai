import unittest
from unittest.mock import AsyncMock
from thinker_ai.agent.memory.brain_memory import BrainMemory
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.agent.provider.llm_schema import Message


class TestBrainMemory(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.brain_memory = BrainMemory(cache_key="test_cache")
        self.mock_llm = AsyncMock(spec=BaseLLM)
        self.mock_llm.aask = AsyncMock(return_value="Summary")

    async def test_add_talk(self):
        msg = Message(content="Hello", role="user")
        self.brain_memory.add_talk(msg)
        self.assertTrue(self.brain_memory.is_dirty)
        self.assertEqual(len(self.brain_memory.history), 1)
        self.assertEqual(self.brain_memory.history[0].content, "Hello")
        self.assertEqual(self.brain_memory.history[0].role, "user")

    async def test_add_answer(self):
        msg = Message(content="Hi there!", role="assistant")
        self.brain_memory.add_answer(msg)
        self.assertTrue(self.brain_memory.is_dirty)
        self.assertEqual(len(self.brain_memory.history), 1)
        self.assertEqual(self.brain_memory.history[0].content, "Hi there!")
        self.assertEqual(self.brain_memory.history[0].role, "assistant")

    async def test_dumps_and_loads(self):
        msg = Message(content="Hello", role="user")
        self.brain_memory.add_talk(msg)
        await self.brain_memory.dumps()
        self.assertIn("test_cache", BrainMemory.cache)

        # 清除实例，重新加载
        loaded_brain_memory = await BrainMemory.loads("test_cache")
        self.assertEqual(len(loaded_brain_memory.history), 1)
        self.assertEqual(loaded_brain_memory.history[0].content, "Hello")

    async def test_summarize(self):
        # 添加多条消息
        msgs = [
            Message(content="Hello", role="user"),
            Message(content="Hi there!", role="assistant"),
            Message(content="How are you?", role="user"),
            Message(content="I'm fine, thanks!", role="assistant")
        ]
        for msg in msgs:
            if msg.role == "user":
                self.brain_memory.add_talk(msg)
            else:
                self.brain_memory.add_answer(msg)

        # 设置 max_words，确保调用 llm.aask
        summary = await self.brain_memory.summarize(llm=self.mock_llm, max_words=10)
        self.assertEqual(summary, "Summary")
        self.mock_llm.aask.assert_called()

    async def test_get_title(self):
        msg = Message(content="This is a conversation.", role="user")
        self.brain_memory.add_talk(msg)
        title = await self.brain_memory.get_title(llm=self.mock_llm, max_words=5)
        self.assertEqual(title, "Summary")
        self.mock_llm.aask.assert_called()

    async def test_exists(self):
        msg = Message(content="Hello", role="user")
        self.brain_memory.add_talk(msg)
        self.assertTrue(self.brain_memory.exists("Hello"))
        self.assertFalse(self.brain_memory.exists("Hi"))

    async def test_save_and_load_cache_to_disk(self):
        msg = Message(content="Hello", role="user")
        self.brain_memory.add_talk(msg)
        await self.brain_memory.dumps()

        # 保存缓存到磁盘
        BrainMemory.save_cache_to_disk("test_cache.json")

        # 清除内存缓存
        BrainMemory.cache = {}

        # 从磁盘加载缓存
        BrainMemory.load_cache_from_disk("test_cache.json")
        self.assertIn("test_cache", BrainMemory.cache)
        loaded_brain_memory = await BrainMemory.loads("test_cache")
        self.assertEqual(len(loaded_brain_memory.history), 1)
        self.assertEqual(loaded_brain_memory.history[0].content, "Hello")

    async def test_rewrite(self):
        sentence = "It's a sunny day."
        context = "The weather is great today."
        self.mock_llm.aask = AsyncMock(return_value="It's a sunny day because the weather is great today.")
        result = await self.brain_memory.rewrite(sentence=sentence, context=context, llm=self.mock_llm)
        self.assertEqual(result, "It's a sunny day because the weather is great today.")

    async def test_is_related(self):
        text1 = "The cat sat on the mat."
        text2 = "A feline rested on a rug."
        self.mock_llm.aask = AsyncMock(return_value="TRUE")
        result = await self.brain_memory.is_related(text1=text1, text2=text2, llm=self.mock_llm)
        self.assertTrue(result)

    async def test_extract_info(self):
        input_string = "[INFO]: This is some information."
        label, content = self.brain_memory.extract_info(input_string)
        self.assertEqual(label, "INFO")
        self.assertEqual(content, "This is some information.")

    async def test_history_text(self):
        msgs = [
            Message(content="Hello", role="user"),
            Message(content="Hi there!", role="assistant")
        ]
        for msg in msgs:
            if msg.role == "user":
                self.brain_memory.add_talk(msg)
            else:
                self.brain_memory.add_answer(msg)
        history_text = self.brain_memory.history_text
        self.assertIn("Hello", history_text)
        self.assertIn("Hi there!", history_text)

    async def test_split_texts(self):
        text = "a" * 5000
        windows = self.brain_memory.split_texts(text, window_size=1000)
        self.assertTrue(all(len(w) <= 1000 for w in windows))
        self.assertTrue(len(windows) > 1)


if __name__ == "__main__":
    unittest.main()
