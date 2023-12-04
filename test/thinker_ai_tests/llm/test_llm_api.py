import asynctest

from thinker_ai.llm import gpt

model = "gpt-4-1106-preview"


class TestTestGPT(asynctest.TestCase):

    async def test_a_generate_stream(self):
        await gpt.a_generate(model,
                             "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？",
                             stream=True)
        await gpt.a_generate(model,
                             "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？",
                             stream=True)

    async def test_a_completion_batch_text(self):
        inputs: dict[str, str] = {}
        inputs["1"] = "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
        inputs["2"] = "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？"
        result: dict[str, str] = await gpt.a_generate_batch(model, inputs)
        self.assertEqual(2, len(result))


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
