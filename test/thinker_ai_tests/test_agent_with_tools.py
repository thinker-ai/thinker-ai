import asynctest

from thinker_ai.actions import ToolsAction


class AgentWithToolsTestCase(asynctest.TestCase):
    agent = ToolsAction()

    async def test_chat_with_function_call(self):
        self.agent.register_langchain_tools(["google-serper"])
        generated_result = await self.agent.act("查询近十年海平面上升速度和全球气温变化数据")
        self.assertIsNotNone(generated_result)
        print(generated_result)


if __name__ == '__main__':
    asynctest.main()
