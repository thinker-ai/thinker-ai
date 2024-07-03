import json
import os
from thinker_ai.status_machine.state_machine import (StateMachineDefinition, Transition, State, StateMachine, ActionFactory, CompositeState, StateType)


class StateMachineBuilder:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(__file__)

    def create_action(self, action_data:dict):
        action_cls = ActionFactory.create_action(action_data)
        return action_cls

    def create_state(self, state_data):
        actions = set(self.create_action(action) for action in state_data.get('actions', [dict]))
        state_type: StateType = state_data['type']
        if 'inner_state_machine' in state_data:
            inner_state_machine = self.create_state_machine(state_data['inner_state_machine'])
            return CompositeState(
                name=state_data['name'],
                actions=actions,
                inner_state_machine=inner_state_machine,
                to_inner_command_name_map=state_data.get('to_inner_command_name_map', {}),
                from_inner_event_name_map=state_data.get('from_inner_event_name_map', {}),
                type=state_type
            )
        return State(name=state_data['name'], actions=actions, type=state_type)

    def create_transition(self, transition_data, states):
        source_state = next((state for state in states if state.name == transition_data['source']), None)
        target_state = next((state for state in states if state.name == transition_data['target']), None)

        if source_state is None or target_state is None:
            raise ValueError("Source or target state not found")

        return Transition(event=transition_data['event'], source=source_state, target=target_state)

    def create_state_machine(self, state_machine_data):
        states = set(self.create_state(state) for state in state_machine_data['states'])
        transitions = set(self.create_transition(transition, states) for transition in state_machine_data['transitions'])
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

    def read_json_file(self, file_path):
        full_path = os.path.join(self.base_path, file_path)
        with open(full_path, 'r') as file:
            return json.load(file)

    def construct_state_machine(self, file_path):
        json_data = self.read_json_file(file_path)
        return self.create_state_machine(json_data)
