import os
import unittest
from thinker_ai.status_machine.state_machine_scenario import StateMachineContextBuilder, ExecutorRegister
from thinker_ai.status_machine.state_machine_scenario_repository import DefaultStateMachineScenarioRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_executors import StartExecutor, MiddleExecutor


class TestSimpleStateMachine(unittest.TestCase):

    def setUp(self):
        ExecutorRegister.register_executor(StartExecutor.get_full_class_name(), StartExecutor)
        ExecutorRegister.register_executor(MiddleExecutor.get_full_class_name(), MiddleExecutor)
        self.base_dir = os.path.dirname(__file__)
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      'test_simple_state_machine_definitions.json')
        self.scenario_repo = DefaultStateMachineScenarioRepository.from_file(self.base_dir,
                                                                           'test_simple_state_machine_scenarios.json',
                                                                             StateMachineContextBuilder(),
                                                                             self.definition_repo)
        # 读取状态机实例
        self.state_machine = self.scenario_repo.get("group_scenario","simple_scenario")

    def tearDown(self):
        self.state_machine = None
        self.state_machine = None

    def test_simple_state_machine(self):
        # Transition to middle state
        commands = self.state_machine.get_state_machine_def().get_self_validate_commands_in_order()
        self.state_machine.handle(commands[0][0])
        self.assertEqual(self.state_machine.current_state_scenario_des.state_def.name, "middle")
        # Transition to end state
        result = self.state_machine.handle(commands[0][1])
        self.assertIsNotNone(result.event)
        self.assertEqual(result.event.name, "middle_command_result_success")
        self.assertEqual(self.state_machine.current_state_scenario_des.state_def.name, "end")
        self.assertEqual(self.state_machine.last_state().state_def.name, "middle")


if __name__ == '__main__':
    unittest.main()
