import unittest

from thinker_ai.agent.memory import Memory


class TestMemory(unittest.TestCase):

    def test_add_and_get(self):
        memory = Memory('agent1')
        memory.add('topic1', 'message1')
        self.assertIn('message1', memory.storage['topic1'])

    def test_add_batch(self):
        memory = Memory('agent1')
        messages = ['message1', 'message2']
        memory.add_batch('topic1', messages)
        self.assertEqual(len(memory.storage['topic1']), 2)
        self.assertIn('message2', memory.storage['topic1'])

    def test_get_by_keyword(self):
        memory = Memory('agent1')
        memory.add('topic1', 'hello world')
        memory.add('topic1', 'hello')
        self.assertEqual(len(memory.get_by_keyword('topic1', 'world')), 1)

    def test_delete(self):
        memory = Memory('agent1')
        memory.add('topic1', 'message1')
        memory.delete('topic1', 'message1')
        self.assertNotIn('message1', memory.storage.get('topic1', []))

    def test_clear(self):
        memory = Memory('agent1')
        memory.add('topic1', 'message1')
        memory.clear()
        self.assertEqual(len(memory.storage), 0)

if __name__ == '__main__':
    unittest.main()
