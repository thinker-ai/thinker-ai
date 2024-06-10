import asynctest
from langchain_core.tools import StructuredTool
from typing import Dict, cast

from langchain.pydantic_v1 import BaseModel

from thinker_ai.agent.provider.llm import open_ai
from thinker_ai.agent.provider.openai_api import FunctionException, OpenAILLM

model = "gpt-4o"

def hello_world(name: str) -> str:
    """
        :param name:str
        :return:str
    """
    return f"{name}是谁？"


class HelloWorldArgs(BaseModel):
    name: str


class TestGPT(asynctest.TestCase):

    def test_generate(self):
        result = open_ai.generate(model,
                              "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
                              "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？")
        self.assertIsNotNone(result)
        print(result)

    async def test_async_generate(self):
        result = await open_ai.a_generate(model,
                                "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
                                "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？")
        self.assertIsNotNone(result)
        print(result)

    def test_generate_stream(self):
        result = open_ai.generate(model,
                              "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
                              "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？",
                              stream=True)
        self.assertIsNotNone(result)
        print(result)

    async def test_a_generate_stream(self):
        result = await open_ai.a_generate(model,
                                      "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？"
                                      "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？",
                                      stream=True)
        self.assertIsNotNone(result)
        print(result)

    def test_function_call(self):
        try:
            StructuredTool.from_function(func=hello_world, args_schema=HelloWorldArgs)
            open_ai.register_function(hello_world, HelloWorldArgs)
            result = open_ai.generate(model, "请调用方法，传入'王立'作为参数，根据返回的结果，回答问题")
            self.assertIsNotNone(result)
            print(result)
        finally:
            open_ai.remove_function("hello_world")

    def test_function_call_with_steam(self):
        try:
            StructuredTool.from_function(func=hello_world, args_schema=HelloWorldArgs)
            open_ai.register_function(hello_world, HelloWorldArgs)
            with self.assertRaises(FunctionException):
                open_ai.generate(model, "请调用方法，传入'王立'作为参数，根据返回的结果，回答问题", stream=True)
        finally:
            open_ai.remove_function("hello_world")

    async def test_async_function_call(self):
        try:
            StructuredTool.from_function(func=hello_world, args_schema=HelloWorldArgs)
            open_ai.register_function(hello_world, HelloWorldArgs)
            with self.assertRaises(FunctionException):
                await open_ai.a_generate(model, "请调用方法，传入'王立'作为参数，根据返回的结果，回答问题")
        finally:
            open_ai.remove_function("hello_world")

    async def test_async_function_call_with_steam(self):
        try:
            StructuredTool.from_function(func=hello_world, args_schema=HelloWorldArgs)
            open_ai.register_function(hello_world, HelloWorldArgs)
            result = await open_ai.a_generate(model, "请调用方法，传入'王立'作为参数，根据返回的结果，回答问题", stream=True)
            self.assertIsNotNone(result)
            print(result)
        finally:
            open_ai.remove_function("hello_world")

    async def test_a_completion_batch_text(self):
        inputs: Dict[str, str] = {
            "user_1": "我用session.request()向你发送消息，你才能记住我们之前交流的上下文是吗，能记住多少条上下文？",
            "2": "如果我不用session.request()向你发送消息，而是每次都新建一个request，你就不能获取我们之前交流的上下文，是吗？"}
        result: Dict[str, str] = await open_ai.a_generate_batch(model, inputs)
        self.assertEqual(2, len(result))

    def test_del_file(self):
        file_dir = "data/test.md"
        file_id = open_ai.upload_file(file_dir,"assistants").id
        deleted = open_ai.delete_file(file_id)
        self.assertTrue(deleted)


if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法
