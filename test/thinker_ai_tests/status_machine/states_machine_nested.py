import unittest
from thinker_ai.status_machine.state_machine import Command, Action, ActionFactory, Event
from thinker_ai_tests.status_machine.state_machine_utils import construct_state_machine


class TestNestedStateMachine(unittest.TestCase):

    def setUp(self):
        self.state_machine = construct_state_machine('states_machine_nested.json')

    def tearDown(self):
        self.state_machine = None

    def test_inner_state_machine_mapping(self):
        # Transition to outer composite state
        self.state_machine.handle(Command(name="start_command", target=self.state_machine.definition.id))
        self.assertEqual(self.state_machine.current_state.name, "outer_composite")

        # Handle outer command which will be translated to middle composite command, then to inner command
        command = Command(name="outer_composite_command", target=self.state_machine.definition.id)
        event = self.state_machine.handle(command)

        self.assertIsNotNone(event)
        self.assertEqual(event.name, "outer_composite_command_handled")
        self.assertEqual(self.state_machine.current_state.name, "end")
        self.assertEqual(self.state_machine.last_state().name, "outer_composite")

    def test_inner_state_machine_direct(self):
        # Directly operate the inner state machine via Command's target
        command = Command(name="inner_start_command", target="inner_test_sm")
        self.state_machine.handle(command)

        # Verify the inner state machine's state
        outer_state = next(state for state in self.state_machine.definition.states if state.type == "start")
        middle_state_machine = outer_state.inner_state_machine
        middle_state = next(state for state in middle_state_machine.definition.states if state.type == "start")
        inner_state_machine = middle_state.inner_state_machine

        self.assertEqual(inner_state_machine.current_state.name, "inner_end")
        self.assertEqual(inner_state_machine.last_state().name, "inner_start")

    def test_combined_state_machine(self):
        # Transition to outer composite state
        self.state_machine.handle(Command(name="start_command", target=self.state_machine.definition.id))
        self.assertEqual(self.state_machine.current_state.name, "outer_composite")

        # Directly operate the inner state machine via Command's target and check cascading
        command = Command(name="inner_start_command", target="inner_test_sm")
        self.state_machine.handle(command)
        command = Command(name="middle_composite_command", target="middle_test_sm")
        self.state_machine.handle(command)

        # Verify middle state machine transitions
        outer_state = next(state for state in self.state_machine.definition.states if state.name == "outer_composite")
        middle_state_machine = outer_state.inner_state_machine
        self.assertEqual(middle_state_machine.current_state.name, "middle_end")
        self.assertEqual(middle_state_machine.last_state().name, "middle_composite")

        # Verify inner state machine transitions
        middle_state = next(
            state for state in middle_state_machine.definition.states if state.type == "start")
        inner_state_machine = middle_state.inner_state_machine
        self.assertEqual(inner_state_machine.current_state.name, "inner_end")
        self.assertEqual(inner_state_machine.last_state().name, "inner_start")

        # Check if the outer state machine has also transitioned
        self.assertEqual(self.state_machine.current_state.name, "end")
        self.assertEqual(self.state_machine.last_state().name, "outer_composite")


if __name__ == '__main__':
    unittest.main()
