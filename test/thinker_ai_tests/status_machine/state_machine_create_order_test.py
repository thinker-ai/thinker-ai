import os
import unittest
from typing import Tuple, List

from thinker_ai.status_machine.state_machine import ActionFactory, BaseStateDefinition, StateMachineDefinition
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import InnerStartAction


class StateMachineCreateOrderTest(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'state_execute_paths_test.json'
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      self.definitions_file_name)
        ActionFactory.register_action(InnerStartAction.get_full_class_name(), InnerStartAction)

    def test_double_layer_state_machine(self):
        state_machine_definition: StateMachineDefinition = self.definition_repo.get_root()
        sorted_states_defs: List[Tuple[str, BaseStateDefinition]] = state_machine_definition.get_state_machine_create_order(
            self.definition_repo)
        sorted_states = []
        for name, _ in sorted_states_defs:
            sorted_states.append(name)
        expected_order = [
            "example_sm.A.4",
            "example_sm.A",
        ]
        self.assertEqual(sorted_states,expected_order)
    def test_get_next_state_machine_to_create(self):
        state_machine_definition: StateMachineDefinition = self.definition_repo.get_root()
        name, _ = state_machine_definition.next_state_machine_to_create(self.definition_repo.get_state_machine_names(),self.definition_repo)
        self.assertEqual(name, "example_sm.A.4")


if __name__ == '__main__':
    unittest.main()
