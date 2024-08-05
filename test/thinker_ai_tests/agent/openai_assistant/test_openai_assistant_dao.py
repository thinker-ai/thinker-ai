import unittest
import os
from unittest.mock import MagicMock, patch

from thinker_ai.agent.openai_assistant.openai_assistant_dao import OpenAiAssistantDAO, OpenAiAssistantPO, ThreadPO


class TestOpenAiAssistantDAO(unittest.TestCase):

    @patch('thinker_ai.agent.openai_assistant.openai_assistant_dao.client')
    @patch('thinker_ai.agent.openai_assistant.openai_assistant_dao.PROJECT_ROOT', os.path.dirname(__file__))
    def setUp(self, mock_client):
        test_file_path = os.path.join(os.path.dirname(__file__), 'test_assistants.json')
        self.dao = OpenAiAssistantDAO(test_file_path)
        self.mock_client = mock_client

    def test_add_assistant(self):
        assistant_po = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_1", topic_threads=[])
        self.dao.add_assistant(assistant_po)
        self.assertEqual(self.dao.get_assistant("assistant_1").assistant_id, "assistant_1")

    def test_get_assistant(self):
        assistant_po = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_1", topic_threads=[])
        self.dao.add_assistant(assistant_po)
        retrieved = self.dao.get_assistant("assistant_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.assistant_id, "assistant_1")

    def test_update_assistant(self):
        assistant_po = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_1", topic_threads=[])
        self.dao.add_assistant(assistant_po)
        updated_po = OpenAiAssistantPO(user_id="user_2", assistant_id="assistant_1", topic_threads=[])
        self.dao.update_assistant(updated_po)
        retrieved = self.dao.get_assistant("assistant_1")
        self.assertEqual(retrieved.user_id, "user_2")

    def test_delete_assistant(self):
        assistant_po = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_1", topic_threads=[])
        self.dao.add_assistant(assistant_po)
        self.dao.delete_assistant("assistant_1")
        self.assertIsNone(self.dao.get_assistant("assistant_1"))

    def test_get_my_assistant_ids(self):
        assistant_po_1 = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_1", topic_threads=[])
        assistant_po_2 = OpenAiAssistantPO(user_id="user_1", assistant_id="assistant_2", topic_threads=[])
        self.dao.add_assistant(assistant_po_1)
        self.dao.add_assistant(assistant_po_2)
        ids = self.dao.get_my_assistant_ids("user_1")
        self.assertListEqual(ids, ["assistant_2", "assistant_1"])


if __name__ == '__main__':
    unittest.main()
