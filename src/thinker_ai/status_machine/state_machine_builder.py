from thinker_ai.status_machine.state_machine import (StateMachineDefinition, Transition,
                                                     State, StateMachine, ActionFactory, CompositeState, StateType
                                                     )


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
