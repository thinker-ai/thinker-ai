import unittest

from thinker_ai.status_machine.state_machine import Command
from thinker_ai_tests.status_machine.state_machine_utils import construct_state_machine


class TestSimpleStateMachine(unittest.TestCase):

    def setUp(self):
        self.state_machine = construct_state_machine('states_machine_simple.json')

    def tearDown(self):
        self.state_machine = None

    def test_simple_state_machine(self):
        # Transition to middle state
        self.state_machine.handle(Command(name="start_command", target=self.state_machine.definition.id))
        self.assertEqual(self.state_machine.current_state.name, "middle")

        # Transition to end state
        command = Command(name="middle_command", target=self.state_machine.definition.id)
        event = self.state_machine.handle(command)

        self.assertIsNotNone(event)
        self.assertEqual(event.name, "middle_command_handled")
        self.assertEqual(self.state_machine.current_state.name, "end")
        self.assertEqual(self.state_machine.last_state().name, "middle")


if __name__ == '__main__':
    unittest.main()
