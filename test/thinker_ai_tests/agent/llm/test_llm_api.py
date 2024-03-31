import asynctest
from langchain_core.tools import StructuredTool

from thinker_ai.agent.llm import gpt
from langchain.pydantic_v1 import BaseModel

from thinker_ai.agent.functions.functions_register import FunctionsRegister

model = "gpt-4-0125-preview"

class TestGPT(asynctest.TestCase):
    def test_generate_stream(self):
        gpt.generate(model,"我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？")
        gpt.generate(model,"如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？")

    async def test_a_generate_stream(self):
        await gpt.a_generate(model,
                             "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？",
                             stream=True)
        await gpt.a_generate(model,
                             "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？",
                             stream=True)



    def test_function_call(self):
        def hello_world(name: str) -> str:
            """
                :param name:str
                :return:str
            """
            return f"Hello, {name}!"

        class HelloWorldArgs(BaseModel):
            name: str

        StructuredTool.from_function(func=hello_world, args_schema=HelloWorldArgs)
        gpt.register_function(hello_world, HelloWorldArgs)
        function_call = gpt._generate_function_call(model, "请使用方法，传入'王立'作为参数",
                                                    candidate_functions=gpt.functions_register.functions_schema)
        if function_call:
            # 假设从某处获得参数，这里直接用字典演示
            function = gpt.functions_register.invoke_function(function_call.name)
            # 调用函数
            result = function.invoke(function_call.arguments)
            self.assertEqual(result, "Hello, 王立!")

    async def test_a_completion_batch_text(self):
        inputs: dict[str, str] = {}
        inputs["1"] = "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
        inputs["2"] = "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？"
        result: dict[str, str] = await gpt.a_generate_batch(model, inputs)
        self.assertEqual(2, len(result))


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
