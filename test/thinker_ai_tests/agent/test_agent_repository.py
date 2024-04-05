import unittest
from thinker_ai.agent.agent_repository import AgentRepository
from thinker_ai.agent.agent import Agent
from thinker_ai.agent.llm import gpt


class TestAgentRepository(unittest.TestCase):
    def setUp(self):
        self.client = gpt.llm  # Replace with actual OpenAI client initialization
        self.assistant_id = "asst_n4kxEAYXlisN7mBa9M6t7PdH"
        self.user_id = 'user_1'
        self.agent_id = '1'

        # Replace with actual retrieval of threads and assistant from the GPT-3 API
        assistant = gpt.llm.beta.assistants.retrieve(self.assistant_id)
        threads = {}  # Dictionary of thread objects

        self.agent = Agent(id=self.agent_id, user_id=self.user_id, assistant=assistant, threads=threads,
                           client=self.client)

        # Get the singleton instance of AgentRepository
        self.agent_repo = AgentRepository.get_instance()
        self.agent_repo.add_agent(self.agent, self.user_id)

    def test_agent_creation_and_retrieval(self):
        # Add the agent
        retrieved_agent = self.agent_repo.get_agent(self.agent_id, self.user_id)

        # Verify the agent is retrieved and the assistant id matches
        self.assertIsNotNone(retrieved_agent)
        self.assertEqual(self.assistant_id, retrieved_agent.assistant.id)

    def test_update_agent(self):
        # Perform an update on the assistant_id
        updated_assistant_id = 'asst_GTRz9wVncAlhYqIRcInU0Bus'
        self.agent.assistant.id = updated_assistant_id

        # Update the agent in the repository
        self.agent_repo.update_agent(self.agent, self.user_id)

        # Retrieve the updated agent and verify the changes
        updated_agent = self.agent_repo.get_agent(self.agent_id, self.user_id)
        self.assertIsNotNone(updated_agent)
        self.assertEqual(updated_assistant_id, updated_agent.assistant.id)

    def tearDown(self):
        # Clean up any changes made to the data
        self.agent_repo.delete_agent(self.agent_id, self.user_id)


if __name__ == "__main__":
    unittest.main()
