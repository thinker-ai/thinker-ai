from typing import Dict, Optional

from thinker_ai.status_machine.state_machine import (Command, StateMachineDefinition, Transition,
                                                     State, Action, Event, StateMachine, ActionFactory, CompositeState)
import json
import unittest
import os


class SampleAction(Action):
    def __init__(self, name: str):
        super().__init__(name)

    def handle(self, command: Command) -> Optional[Event]:
        if command.name == self.name:
            return Event(name=f"{self.name}_handled", id=command.id, payload=command.payload)
        return None

    @classmethod
    def from_dict(cls, data: Dict) -> 'SampleAction':
        return cls(name=data['name'])


ActionFactory.register_action('SampleAction', SampleAction)


def read_json_file(file_path):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, file_path)
    with open(full_path, 'r') as file:
        return json.load(file)


def create_action(action_data):
    action_cls = ActionFactory.create_action('SampleAction', action_data)
    return action_cls


def create_state(state_data):
    actions = set(create_action(action) for action in state_data.get('actions', []))
    if 'inner_state_machine' in state_data:
        inner_state_machine = create_state_machine(state_data['inner_state_machine'])
        return CompositeState(
            name=state_data['name'],
            actions=actions,
            inner_state_machine=inner_state_machine,
            to_inner_command_name_map=state_data.get('to_inner_command_name_map', {}),
            from_inner_event_name_map=state_data.get('from_inner_event_name_map', {})
        )
    return State(name=state_data['name'], actions=actions)


def create_transition(transition_data, states):
    source_state = next((state for state in states if state.name == transition_data['source']), None)
    target_state = next((state for state in states if state.name == transition_data['target']), None)

    if source_state is None or target_state is None:
        raise ValueError("Source or target state not found")

    return Transition(event=transition_data['event'], source=source_state, target=target_state)


def create_state_machine(state_machine_data):
    states = set(create_state(state) for state in state_machine_data['states'])
    transitions = set(create_transition(transition, states) for transition in state_machine_data['transitions'])
    definition = StateMachineDefinition(
        id=state_machine_data['definition_id'],
        states=states,
        transitions=transitions
    )
    start_state = next((state for state in states if (state.name == 'start' or state.name == 'inner_start')), None)

    if start_state is None:
        raise ValueError(f"Start state not found in state machine with ID: {state_machine_data['id']}")

    instance_id = state_machine_data['id']
    return StateMachine(definition=definition, id=instance_id, current_state=start_state)


def construct_state_machine(file_path):
    json_data = read_json_file(file_path)
    return create_state_machine(json_data)


class TestSimpleStateMachine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.state_machine = construct_state_machine('states_machine_simple.json')

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
