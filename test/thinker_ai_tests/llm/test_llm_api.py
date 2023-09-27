import asynctest

from thinker_ai.llm.gpt import GPT
from thinker_ai.llm.llm_factory import get_llm

class TestTestGPT(asynctest.TestCase):
    async def test_a_generate_stream(self):
        llm = get_llm()
        await llm.a_generate(
            "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？",stream=True)
        await llm.a_generate(
            "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？",stream=True)

    async def test_a_completion_batch_text(self):
        llm = get_llm()
        inputs:dict[str,str]= {}
        inputs["1"]="我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
        inputs["2"]="如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？"
        result:dict[str,str] = await llm.a_generate_batch(inputs)
        self.assertEqual(2,len(result))

if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
