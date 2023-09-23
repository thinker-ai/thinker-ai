import asynctest

from thinker_ai.llm.gpt import GPT
from thinker_ai.llm.llm_factory import get_llm

class TestTestGPT(asynctest.TestCase):
    async def test_a_generate_stream(self):
        llm = get_llm()
        result = await llm.a_generate_stream("你好")
        self.assertIsNotNone(result)


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
