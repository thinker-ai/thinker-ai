import unittest
from unittest.mock import patch, MagicMock
from thinker_ai.agent.openai_assistant.openai_assistant_repository import OpenAiAssistantRepository, OpenAiAssistant


class TestOpenAiAssistantRepository(unittest.TestCase):

    def setUp(self):
        self.user_id = 'user_1'
        self.assistant_id = 'user_1'
        self.assistant_id = "asst_zBrqXNoQIvnX1TyyVry9UveZ"
        self.assistant = OpenAiAssistant.from_id(user_id=self.user_id, assistant_id=self.assistant_id)

        # Get the singleton instance of AgentRepository
        self.assistant_repo = OpenAiAssistantRepository.get_instance()
        self.assistant_repo.add_assistant(self.assistant, self.user_id)

    def test_assistant_creation_and_retrieval(self):
        # Add the assistant
        retrieved_assistant = self.assistant_repo.get_assistant(self.assistant_id, self.user_id)

        # Verify the assistant is retrieved and the assistant id matches
        self.assertIsNotNone(retrieved_assistant)
        self.assertEqual(self.assistant_id, retrieved_assistant.id)

    def test_update_assistant(self):
        # Perform an update on the assistant_id
        updated_assistant_id = 'asst_GTRz9wVncAlhYqIRcInU0Bus'
        self.assistant.assistant.id = updated_assistant_id

        # Update the assistant in the repository
        self.assistant_repo.update_assistant(self.assistant, self.user_id)

        # Retrieve the updated assistant and verify the changes
        updated_assistant = self.assistant_repo.get_assistant(self.assistant_id, self.user_id)
        self.assertIsNotNone(updated_assistant)
        self.assertEqual(updated_assistant_id, updated_assistant.id)

    def tearDown(self):
        # Clean up any changes made to the data
        self.assistant_repo.delete_assistant(self.assistant_id, self.user_id)


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
