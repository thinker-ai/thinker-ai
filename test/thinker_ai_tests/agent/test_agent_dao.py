import unittest
from unittest.mock import patch

from thinker_ai.agent.agent_dao import AgentDAO, AgentPO


class TestAgentDAO(unittest.TestCase):
    db_path:str='data/test_agents.json'
    def setUp(self):
        # 在测试前准备环境，比如创建一个测试用的文件路径
        self.agent_dao = AgentDAO(filepath=self.db_path)
        self.test_agent = AgentPO(id='user_1', user_id='user_1', threads=[], assistant_id='assistant_1')

    def tearDown(self):
        # 测试后清理环境，例如删除测试文件
        import os
        os.remove(self.db_path)

    def test_add_and_get_agent(self):
        self.agent_dao.add_agent(self.test_agent)
        retrieved_agent = self.agent_dao.get_agent(self.test_agent.id)
        my_agents = self.agent_dao.get_my_agent_ids('user_1')
        self.assertEqual(self.test_agent.id, retrieved_agent.id)
        self.assertEqual(self.test_agent.user_id, retrieved_agent.user_id)
        self.assertEqual(len(my_agents), 1)

    def test_update_agent(self):
        self.agent_dao.add_agent(self.test_agent)
        updated_agent = AgentPO(id='user_1', user_id='user_2', threads=[], assistant_id='assistant_2')
        self.agent_dao.update_agent(updated_agent)
        retrieved_agent = self.agent_dao.get_agent(self.test_agent.id)
        self.assertEqual(updated_agent.user_id, retrieved_agent.user_id)

    def test_delete_agent(self):
        self.agent_dao.add_agent(self.test_agent)
        self.agent_dao.delete_agent(self.test_agent.id)
        self.assertIsNone(self.agent_dao.get_agent(self.test_agent.id))


if __name__ == '__main__':
    unittest.main()
