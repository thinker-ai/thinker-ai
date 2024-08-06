import unittest
import os
from unittest.mock import MagicMock, patch

from thinker_ai.agent.openai_assistant_api.openai_assistant_api_dao import OpenAiAssistantApiDAO, OpenAiAssistantApiPO


class TestOpenAiAssistantDAO(unittest.TestCase):

    @patch('thinker_ai.agent.openai_assistant_api.openai_assistant_api.openai_client')
    @patch('thinker_ai.configs.const.PROJECT_ROOT', new=os.path.dirname(__file__))
    def setUp(self, mock_client):
        test_file_path = os.path.join(os.path.dirname(__file__), 'test_assistants.json')
        self.dao = OpenAiAssistantApiDAO(test_file_path)
        self.mock_client = mock_client

    def test_add_assistant(self):
        assistant_po = OpenAiAssistantApiPO(user_id="user_1",name="assistant_name", assistant_id="assistant_1", topic_threads={})
        self.dao.add_assistant_api(assistant_po)
        self.assertEqual(self.dao.get_assistant_api_by_id("assistant_1").assistant_id, "assistant_1")

    def test_get_assistant(self):
        assistant_po = OpenAiAssistantApiPO(user_id="user_1", name="assistant_name", assistant_id="assistant_1", topic_threads={})
        self.dao.add_assistant_api(assistant_po)
        retrieved = self.dao.get_assistant_api_by_id("assistant_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.assistant_id, "assistant_1")

    def test_update_assistant_name(self):
        assistant_po = OpenAiAssistantApiPO(user_id="user_1",name="old_name", assistant_id="assistant_1", topic_threads={})
        self.dao.add_assistant_api(assistant_po)
        self.dao.update_assistant_name(user_id="user_1",assistant_id="assistant_1",name="new_name")
        retrieved = self.dao.get_assistant_api_by_id("assistant_1")
        self.assertEqual(retrieved.name, "new_name")

    def test_update_topic_name(self):
        assistant_po = OpenAiAssistantApiPO(user_id="user_1",name="old_name", assistant_id="assistant_1", topic_threads={"old_topic":"123"})
        self.dao.add_assistant_api(assistant_po)
        self.dao.update_topic_name(user_id="user_1",assistant_id="assistant_1",old_topic="old_topic",new_topic="new_topic")
        retrieved = self.dao.get_assistant_api_by_id("assistant_1")
        self.assertEqual(retrieved.topic_threads.get("new_topic"), "123")

    def test_delete_assistant(self):
        assistant_po = OpenAiAssistantApiPO(user_id="user_1", name="assistant_name",assistant_id="assistant_1", topic_threads={})
        self.dao.add_assistant_api(assistant_po)
        self.dao.delete_assistant_api("assistant_1")
        self.assertIsNone(self.dao.get_assistant_api_by_id("assistant_1"))

    def test_get_my_assistant_ids(self):
        assistant_po_1 = OpenAiAssistantApiPO(user_id="user_1",name="assistant_name", assistant_id="assistant_1", topic_threads={})
        assistant_po_2 = OpenAiAssistantApiPO(user_id="user_1",name="assistant_name", assistant_id="assistant_2", topic_threads={})
        self.dao.add_assistant_api(assistant_po_1)
        self.dao.add_assistant_api(assistant_po_2)
        ids = self.dao.get_my_assistants_of("user_1", "assistant_id")
        self.assertCountEqual(ids, ["assistant_2", "assistant_1"])


if __name__ == '__main__':
    unittest.main()
