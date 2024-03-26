import asynctest

from thinker_ai.agent.function_call_agent import FunctionCallAgent


class AgentWithToolsTestCase(asynctest.TestCase):
    agent = FunctionCallAgent()

    async def test_chat_with_function_call(self):
        self.agent.register_langchain_functions(["google-serper"])
        generated_result = await self.agent.act("gpt-4-0125-preview","查询近十年海平面上升速度和全球气温变化数据")
        self.assertIsNotNone(generated_result)
        print(generated_result)


if __name__ == '__main__':
    asynctest.main()
