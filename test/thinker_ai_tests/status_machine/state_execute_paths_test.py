import os
import unittest
from typing import List, Tuple

from thinker_ai.status_machine.state_machine import StateMachineDefinition, BaseStateDefinition
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository


class TestStateMachineDefinition(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'test_execute_plan.json'
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      self.definitions_file_name)

    def test_get_state_execute_paths(self):
        state_machine_definition: StateMachineDefinition = self.definition_repo.get_root()
        execute_paths:List[List[Tuple[str, BaseStateDefinition]]] = state_machine_definition.get_state_validate_paths()
        actual_paths=[]
        for path in execute_paths:
            plan = []
            for name,_ in path:
                plan.append(name)
            actual_paths.append(plan)

        expected_path = [[
            "example_sm.start",
            "example_sm.A",
            "example_sm.B",
            "example_sm.C",
            "example_sm.A",
            "example_sm.B",
            "example_sm.D",
            "example_sm.E",
        ],['example_sm.start',
           'example_sm.A',
           'example_sm.B',
           'example_sm.D',
           'example_sm.E']
        ,['example_sm.start',
          'example_sm.A',
          'example_sm.B',
          'example_sm.C',
          'example_sm.A',
          'example_sm.B']
        ]
        for plan in actual_paths:
            self.assertTrue(plan in expected_path)


if __name__ == '__main__':
    unittest.main()
