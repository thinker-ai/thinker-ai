import unittest
from pprint import pprint
from typing import List, Dict, Any

from thinker_ai.agent.agent_repository import AgentRepository
from thinker_ai.main import create_agent, del_agent, upload_file, delete_file, ask

user_id = "test_user"


def setup_customer_support_agent() -> str:
    agent_name = "客户支持"
    instructions = "你是一个客户支持聊天机器人。使用您的知识库，以最佳方式回复客户询问。"
    tools = [{"type": "retrieval"}]
    file_ids = []
    agent_id = create_agent("gpt-4-0125-preview", user_id, agent_name, instructions, tools, file_ids)
    return agent_id


def setup_math_teacher_agent() -> str:
    agent_name = "数学辅导员"
    instructions = "你是一名私人数学辅导员。当被问到数学问题时，编写并运行代码来回答问题。"
    tools = [{"type": "code_interpreter"}]
    file_ids = []
    agent_id = create_agent("gpt-4-0125-preview", user_id, agent_name, instructions, tools, file_ids)
    return agent_id


def teardown_agent(agent_id: str) -> bool:
    result = del_agent(user_id, agent_id)
    return result


class MainTest(unittest.TestCase):

    def test_teardown_agent(self):
        agent_id = setup_customer_support_agent()
        agent = AgentRepository.get_agent(user_id, agent_id)
        self.assertIsNotNone(agent)
        deleted = teardown_agent(agent_id)
        self.assertTrue(deleted)
        agent = AgentRepository.get_agent(user_id, agent_id)
        self.assertIsNone(agent)

    def test_del_file(self):
        file_dir = "../test.md"
        file_id = upload_file(file_dir)
        deleted = delete_file(file_id)
        self.assertTrue(deleted)

    def test_ask_for_customer_support(self):
        agent_id = setup_customer_support_agent()
        try:
            results: Dict[str, Any] = ask(user_id=user_id, agent_name="客户支持",topic="养生",
                                          content="吃人参和灵芝有什么不同？")
            self.do_with_results(results)
        finally:
            teardown_agent(agent_id)

    def test_ask_math_teacher(self):
        agent_id = setup_math_teacher_agent()
        try:
            results: Dict[str, Any] = ask(user_id=user_id, agent_name="数学辅导员",topic="初二数学",
                                          content="请用一个坐标图表示一个一元二次方程")
            self.do_with_results(results)
        finally:
            teardown_agent(agent_id)

    def do_with_results(self, results):
        self.assertGreater(len(results), 0)
        for key in results.keys():
            if key == "text":
                pprint(results[key])
            if key == "image_file":
                with open(f"../my-image.png", "wb") as file:
                    file.write(results[key])


if __name__ == '__main__':
    unittest.main()
