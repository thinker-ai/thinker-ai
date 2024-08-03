import unittest
import os

from thinker_ai.status_machine.state_machine_scenario import StateMachineContextBuilder, ExecutorRegister
from thinker_ai.status_machine.state_machine_scenario_repository import DefaultStateMachineScenarioRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_executors import InnerStartExecutor, MiddleStartAction, OuterStartAction


class TestNestedStateMachine(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'test_nested_state_machine_definitions.json'
        self.scenarios_file_name = 'test_nested_state_machine_scenarios.json'
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      self.definitions_file_name)
        self.scenario_repo = DefaultStateMachineScenarioRepository.from_file(self.base_dir, self.scenarios_file_name,
                                                                             StateMachineContextBuilder(),
                                                                             self.definition_repo)
        ExecutorRegister.register_executor(InnerStartExecutor.get_full_class_name(), InnerStartExecutor)
        ExecutorRegister.register_executor(MiddleStartAction.get_full_class_name(), MiddleStartAction)
        ExecutorRegister.register_executor(OuterStartAction.get_full_class_name(), OuterStartAction)
        # 读取状态机实例
        self.state_machine = self.scenario_repo.get("group_scenario","outer_scenario")

    def tearDown(self):
        self.state_machine = None

    def test_combined_state_machine(self):
        commands = self.state_machine.get_state_machine_def().get_self_validate_commands_in_order()
        self.state_machine.handle(commands[0][0])
        # Verify middle state machine transitions
        middle_state_machine = self.scenario_repo.get("group_scenario","outer_start_scenario")
        self.assertEqual("middle_end", middle_state_machine.current_state_scenario_des.state_def.name)
        self.assertEqual("middle_start", middle_state_machine.last_state().state_def.name)
        # Check if the outer state machine has also transitioned
        self.assertEqual("outer_end", self.state_machine.current_state_scenario_des.state_def.name)
        self.assertEqual("outer_start", self.state_machine.last_state().state_def.name)


if __name__ == '__main__':
    unittest.main()
