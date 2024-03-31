import unittest
from unittest.mock import MagicMock, patch

from thinker_ai.agent.agent_dao import AgentDAO, AgentPO
from thinker_ai.agent.agent_repository import AgentRepository
from thinker_ai.agent.llm import gpt


class TestAgentRepository(unittest.TestCase):
    def setUp(self):
        self.mock_dao = MagicMock(spec=AgentDAO)
        self.agent_repo = AgentRepository(agent_dao=self.mock_dao, client=gpt.llm)
        self.assistant_id = "asst_n4kxEAYXlisN7mBa9M6t7PdH"
        self.threads = {}
        self.test_agent_po = AgentPO(id='1', user_id='user_1', assistant_id=self.assistant_id, threads=self.threads)

    def _add_mock_agent(self, user_id='user_1', agent_id='1', assistant_id="asst_n4kxEAYXlisN7mBa9M6t7PdH"):
        mock_agent = MagicMock(user_id=user_id, id=agent_id)
        mock_agent.assistant.id = assistant_id
        mock_agent.threads.values.return_value = self.threads
        self.mock_dao.get_agent.return_value = self.test_agent_po
        self.agent_repo.add_agent(mock_agent, user_id)
        return mock_agent

    @patch('thinker_ai.agent.llm.gpt.llm.beta.assistants.retrieve')
    def test_add_and_get_agent(self, mock_retrieve_assistant):
        mock_retrieve_assistant.return_value = MagicMock()
        self._add_mock_agent()

        retrieved_agent = self.agent_repo.get_agent('1', 'user_1')
        self.assertIsNotNone(retrieved_agent)
        self.assertEqual('1', retrieved_agent.id)
        self.assertEqual('user_1', retrieved_agent.user_id)

    def test_add_and_update_agent(self):
        mock_agent = self._add_mock_agent()
        updated_assistant_id = self.assistant_id
        mock_agent.assistant.id = "updated_assistant_id"
        self.test_agent_po.assistant_id = updated_assistant_id
        self.mock_dao.get_agent.return_value = self.test_agent_po

        self.agent_repo.update_agent(mock_agent, 'user_1')
        updated_agent = self.agent_repo.get_agent('1', 'user_1')

        self.assertEqual(updated_assistant_id, updated_agent.assistant.id)

    def test_add_and_delete_agent(self):
        self._add_mock_agent()
        self.agent_repo.delete_agent('1', 'user_1')

        self.mock_dao.delete_agent.assert_called_once_with('1')
        # 显式设置get_agent在删除操作后返回None
        self.mock_dao.get_agent.return_value = None
        retrieved_after_delete = self.agent_repo.get_agent('1', 'user_1')
        self.assertIsNone(retrieved_after_delete)


if __name__ == '__main__':
    unittest.main()
