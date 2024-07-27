import unittest

from thinker_ai.status_machine.state_machine import StateDefinition, Transition, StateMachineDefinition, \
    get_state_execute_order


class TestStateMachineDefinition(unittest.TestCase):
    def test_get_state_execute_order(self):
        # 创建一个单层示例状态机
        start = StateDefinition("start", "Start", "Start state", "task", "StateContext", set(), set(), is_start=True)
        state_a = StateDefinition("A", "State A", "State A description", "task", "StateContext", set(), set())
        state_b = StateDefinition("B", "State B", "State B description", "task", "StateContext", set(), set())
        state_c = StateDefinition("C", "State C", "State C description", "task", "StateContext", set(), set())
        state_d = StateDefinition("D", "State D", "State D description", "task", "StateContext", set(), set())
        state_e = StateDefinition("E", "State E", "State E description", "task", "StateContext", set(), set())

        transitions = {
            Transition("start_to_A", start, state_a),
            Transition("A_to_B", state_a, state_b),
            Transition("B_to_C", state_b, state_c),
            Transition("C_to_A", state_c, state_a),
            Transition("B_to_D", state_b, state_d),
            Transition("D_to_E", state_d, state_e)
        }

        state_machine_def = StateMachineDefinition("example_sm", "Example State Machine", {start, state_a, state_b, state_c, state_d, state_e}, transitions)

        sorted_states = get_state_execute_order(state_machine_def)
        expected_order_1 = [
            "example_sm.start",
            "example_sm.A",
            "example_sm.B",
            "example_sm.C",
            "example_sm.D",
            "example_sm.E"
        ]
        expected_order_2 = [
            "example_sm.start",
            "example_sm.A",
            "example_sm.B",
            "example_sm.D",
            "example_sm.E",
            "example_sm.C"
        ]
        try:
            self.assertEqual(list(sorted_states), expected_order_1)
        except Exception:
            self.assertEqual(list(sorted_states), expected_order_2)

    def test_double_layer_state_machine(self):
        # 创建一个双层示例状态机
        start = StateDefinition("start", "Start", "Start state", "task", "StateContext", set(), set(), is_start=True)
        state_a = StateDefinition("A", "State A", "State A description", "task", "StateContext", set(), set())
        state_c = StateDefinition("C", "State C", "State C description", "task", "StateContext", set(), set())
        state_d = StateDefinition("D", "State D", "State D description", "task", "StateContext", set(), set())
        state_e = StateDefinition("E", "State E", "State E description", "task", "StateContext", set(), set())

        sub_start = StateDefinition("sub_start", "Sub Start", "Sub start state", "task", "StateContext", set(), set(),
                                    is_start=True)
        sub_state_x = StateDefinition("X", "State X", "State X description", "task", "StateContext", set(), set())
        sub_state_y = StateDefinition("Y", "State Y", "State Y description", "task", "StateContext", set(), set())

        sub_transitions = {
            Transition("sub_start_to_X", sub_start, sub_state_x),
            Transition("X_to_Y", sub_state_x, sub_state_y)
        }

        sub_state_machine_def = StateMachineDefinition("sub_example_sm", "Sub Example State Machine",
                                                       {sub_start, sub_state_x, sub_state_y}, sub_transitions)

        state_with_sub_state_machine = StateDefinition("B", "State B", "State B description", "task", "StateContext",
                                                       set(), set(),
                                                       inner_state_machine_definition=sub_state_machine_def)

        transitions_with_sub = {
            Transition("start_to_A", start, state_a),
            Transition("A_to_B", state_a, state_with_sub_state_machine),
            Transition("B_to_C", state_with_sub_state_machine, state_c),
            Transition("C_to_A", state_c, state_a),
            Transition("B_to_D", state_with_sub_state_machine, state_d),
            Transition("D_to_E", state_d, state_e)
        }

        state_machine_def_with_sub = StateMachineDefinition("example_sm_with_sub", "Example State Machine with Sub",
                                                            {start, state_a, state_with_sub_state_machine, state_c,
                                                             state_d, state_e}, transitions_with_sub)

        sorted_states = get_state_execute_order(state_machine_def_with_sub)
        expected_order_1= [
             'sub_example_sm.sub_start',
             'sub_example_sm.X',
             'sub_example_sm.Y',
             'example_sm_with_sub.start',
             'example_sm_with_sub.A',
             'example_sm_with_sub.B',
             'example_sm_with_sub.C',
             'example_sm_with_sub.D',
             'example_sm_with_sub.E'
        ]
        expected_order_2 = [
             'sub_example_sm.sub_start',
             'sub_example_sm.X',
             'sub_example_sm.Y',
             'example_sm_with_sub.start',
             'example_sm_with_sub.A',
             'example_sm_with_sub.B',
             'example_sm_with_sub.E',
             'example_sm_with_sub.C',
             'example_sm_with_sub.D'
        ]
        try:
            self.assertEqual(sorted_states, expected_order_1)
        except Exception:
            self.assertEqual(sorted_states, expected_order_2)