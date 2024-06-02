import os
import unittest

from thinker_ai.agent.memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):

    def setUp(self):
        self.file_path = 'test_memory.q'
        self.memory = LongTermMemory('agent1', self.file_path)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_file_save_and_load(self):
        self.memory.add('topic1', 'message1')
        del self.memory  # Force deletion to simulate program restart
        new_memory = LongTermMemory('agent1', self.file_path)
        self.assertIn('message1', new_memory.storage.get('topic1', []))

    def test_add_batch_and_persistence(self):
        messages = ['message1', 'message2']
        self.memory.add_batch('topic1', messages)
        del self.memory
        new_memory = LongTermMemory('agent1', self.file_path)
        self.assertEqual(len(new_memory.storage.get('topic1', [])), 2)

if __name__ == '__main__':
    unittest.main()