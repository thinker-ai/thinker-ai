import unittest

from thinker_ai.task import Actor
from thinker_ai.role.agent_with_tools import AgentWithTools
from thinker_ai.context import Context


class AgentWithToolsTestCase(unittest.TestCase):
    context = Context(organization_id="xx", solution_name="climate_change_prediction", actor=Actor("any", "role_x"))
    agent = AgentWithTools(context)

    def test_chat_with_function_call(self):
        self.agent.register_langchain_tools(["google-serper"])
        generated_result = self.agent.chat_with_function_call("查询近十年海平面上升速度和全球气温变化数据")
        self.assertIsNotNone(generated_result)
        print(generated_result)


if __name__ == '__main__':
    unittest.main()
