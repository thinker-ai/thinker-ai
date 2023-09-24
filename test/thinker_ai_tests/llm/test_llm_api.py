import asynctest

from thinker_ai.llm.gpt import GPT
from thinker_ai.llm.llm_factory import get_llm

class TestTestGPT(asynctest.TestCase):
    async def test_a_generate_stream(self):
        llm = get_llm()
        await llm.a_generate_stream(
            "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？")
        await llm.a_generate_stream(
            "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？")


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
