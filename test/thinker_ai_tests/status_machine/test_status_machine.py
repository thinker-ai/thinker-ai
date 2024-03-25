import unittest
from typing import Optional, Dict

from thinker_ai.status_machine.state_machine import Action, Command, Event, ActionFactory, State, \
    StateMachineDefinition, StateMachine, Transition


class TestAction(Action):
    def handle(self, command: Command) -> Optional[Event]:
        if command.name == "test_command":
            return Event(name="test_event")
        return None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        return cls()


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        ActionFactory.register_action("test_action", TestAction)

    def test_handle_valid_command(self):
        start_state = State(name="start", actions={"test_command": TestAction()})
        end_state = State(name="end", actions={})
        transition = Transition(event="test_event", source=start_state, target=end_state)

        sm_definition = StateMachineDefinition(
            business_id="test_business",
            states={"start": start_state, "end": end_state},
            transitions={"test_event": [transition]}
        )

        sm = StateMachine(definition=sm_definition, instance_id="1", current_state=start_state, history=[])
        event = sm.handle(Command(name="test_command"))
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "test_event")

    def test_no_action_for_command(self):
        state = State(name="start", actions={})
        sm_definition = StateMachineDefinition(business_id="test_business", states={"start": state}, transitions={})
        sm = StateMachine(definition=sm_definition, instance_id="1", current_state=state, history=[])

        event = sm.handle(Command(name="test_command"))

        self.assertIsNone(event)

    def test_transition_on_event(self):
        start_state = State(name="start", actions={"test_command": TestAction()})
        end_state = State(name="end", actions={})
        transition = Transition(event="test_event", source=start_state, target=end_state)

        sm_definition = StateMachineDefinition(business_id="test_business", states={
            "start": start_state, "end": end_state}, transitions={"test_event": [transition]})

        sm = StateMachine(definition=sm_definition, instance_id="1", current_state=start_state, history=[])

        sm.handle(Command(name="test_command"))

        self.assertEqual(sm.current_state.name, "end")
        self.assertEqual(sm.last_state().name, "start")

    def test_no_transition_for_event(self):
        state = State(name="start", actions={"test_command": TestAction()})
        sm_definition = StateMachineDefinition(business_id="test_business", states={"start": state}, transitions={})
        sm = StateMachine(definition=sm_definition, instance_id="1", current_state=state, history=[])

        with self.assertRaises(ValueError):
            sm.handle(Command(name="test_command"))

    # You can continue adding more test cases as needed.


if __name__ == "__main__":
    unittest.main()
