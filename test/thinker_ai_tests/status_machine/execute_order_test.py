import os
import unittest
from typing import Tuple, List

from thinker_ai.status_machine.state_machine import ActionFactory, BaseStateDefinition
from thinker_ai.status_machine.status_machine_definition_repository import FileBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import InnerStartAction


class TestStateMachineDefinition(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'test_execute_order_state_machine_definitions.json'
        self.definition_repo = FileBasedStateMachineDefinitionRepository(self.base_dir, self.definitions_file_name)
        ActionFactory.register_action(InnerStartAction.get_full_class_name(), InnerStartAction)

    def test_double_layer_state_machine(self):
        sorted_states_defs: List[Tuple[str, BaseStateDefinition]] = self.definition_repo.get_state_execute_order("example_sm")
        sorted_states = []
        for name, _ in sorted_states_defs:
            sorted_states.append(name)
        expected_order_1 = [
            "example_sm.A.start",
            "example_sm.A.1",
            "example_sm.A.2",
            "example_sm.A.3",
            "example_sm.A.4",
            "example_sm.A.5",
            'example_sm.start',
            "example_sm.A",
            "example_sm.B",
            "example_sm.C",
            "example_sm.D",
            "example_sm.E"
        ]
        expected_order_2 = [
            "example_sm.A.start",
            "example_sm.A.1",
            "example_sm.A.2",
            "example_sm.A.4",
            "example_sm.A.5",
            "example_sm.A.3",
            'example_sm.start',
            "example_sm.A",
            "example_sm.B",
            "example_sm.C",
            "example_sm.D",
            "example_sm.E"
        ]
        expected_order_3 = [
            "example_sm.A.start",
            "example_sm.A.1",
            "example_sm.A.2",
            "example_sm.A.4",
            "example_sm.A.5",
            "example_sm.A.3",
            'example_sm.start',
            "example_sm.A",
            "example_sm.B",
            "example_sm.D",
            "example_sm.E",
            "example_sm.C"
        ]
        expected_order_4 = [
            "example_sm.A.start",
            "example_sm.A.1",
            "example_sm.A.2",
            "example_sm.A.3",
            "example_sm.A.4",
            "example_sm.A.5",
            'example_sm.start',
            "example_sm.A",
            "example_sm.B",
            "example_sm.D",
            "example_sm.E",
            "example_sm.C"
        ]
        try:
            self.assertEqual(expected_order_1,sorted_states)
        except Exception:
            try:
                self.assertEqual(expected_order_2,sorted_states)
            except Exception:
                try:
                    self.assertEqual(expected_order_3,sorted_states)
                except Exception:
                    self.assertEqual(expected_order_4,sorted_states)

    def test_get_next_state_machine_to_create(self):
        name, _ = self.definition_repo.next_state_machine_to_create()
        self.assertEqual(name, "example_sm.A.4")

if __name__ == '__main__':
    unittest.main()
