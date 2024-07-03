from typing import Dict, Optional, Set, Literal

from thinker_ai.status_machine.state_machine import (Command, StateMachineDefinition, Transition,
                                                     State, Action, Event, StateMachine, ActionFactory, CompositeState)
import json
import unittest
import os

StateType = Literal["start", "middle", "end"]


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
    state_type: StateType = state_data['type']
    if 'inner_state_machine' in state_data:
        inner_state_machine = create_state_machine(state_data['inner_state_machine'])
        return CompositeState(
            name=state_data['name'],
            actions=actions,
            inner_state_machine=inner_state_machine,
            to_inner_command_name_map=state_data.get('to_inner_command_name_map', {}),
            from_inner_event_name_map=state_data.get('from_inner_event_name_map', {}),
            type=state_type
        )
    return State(name=state_data['name'], actions=actions, type=state_type)


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

    start_state = next((state for state in states if state.type == 'start'), None)

    if start_state is None:
        raise ValueError(
            f"Start state not found in state machine with definition ID: {state_machine_data['definition_id']}")

    return StateMachine(definition=definition, id=state_machine_data['id'], current_state=start_state)


class TestNestedStateMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_machine_data = read_json_file('states_machine_nested.json')

    def setUp(self):
        # 每个测试方法前都会重新创建一个状态机实例
        self.state_machine = create_state_machine(self.state_machine_data)

    def tearDown(self):
        # 在每个测试方法后清理状态
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
