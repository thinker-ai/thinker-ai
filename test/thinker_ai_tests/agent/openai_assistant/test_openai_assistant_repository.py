import unittest
from thinker_ai.agent.openai_assistant_api.openai_assistant_api_repository import OpenAiAssistantRepository, \
    OpenAiAssistantApi


class TestOpenAiAssistantRepository(unittest.TestCase):

    def setUp(self):
        self.user_id = 'user_1'
        self.assistant_id = "asst_zBrqXNoQIvnX1TyyVry9UveZ"
        self.assistant = OpenAiAssistantApi.from_id(user_id=self.user_id, assistant_id=self.assistant_id)

        # Get the singleton instance of AgentRepository
        self.assistant_repo = OpenAiAssistantRepository.get_instance()
        self.assistant_repo.add_assistant_api(self.user_id, self.assistant)

    def test_assistant_creation_and_retrieval(self):
        # Add the assistant
        retrieved_assistant = self.assistant_repo.get_assistant_api_by_id(user_id=self.user_id,
                                                                          assistant_id=self.assistant_id)

        # Verify the assistant is retrieved and the assistant id matches
        self.assertIsNotNone(retrieved_assistant)
        self.assertEqual(self.assistant_id, retrieved_assistant.id)
    def test_get_assistant_api_by_name(self):
        # Add the assistant
        retrieved_assistant = self.assistant_repo.get_assistant_api_by_name(user_id=self.user_id,
                                                                            name="old_assistant")
        # Verify the assistant is retrieved and the assistant id matches
        self.assertIsNotNone(retrieved_assistant)
        self.assertEqual(self.assistant_id, retrieved_assistant.id)

    def test_update_assistant_name(self):
        assistant_name = 'new_assistant'
        # Update the assistant in the repository
        retrieved_assistant = self.assistant_repo.get_assistant_api_by_name(user_id=self.user_id,
                                                                            name="old_assistant")
        retrieved_assistant.name=assistant_name
        self.assistant_repo.update(self.user_id,retrieved_assistant)

        # Retrieve the updated assistant and verify the changes
        updated_assistant = self.assistant_repo.get_assistant_api_by_id(self.user_id, self.assistant_id)
        self.assertIsNotNone(updated_assistant)
        self.assertEqual(assistant_name, updated_assistant.name)

    def tearDown(self):
        # Clean up any changes made to the data
        self.assistant_repo.delete_assistant_api(self.user_id, self.assistant_id)


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
