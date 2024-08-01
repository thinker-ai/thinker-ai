import os
import unittest
from thinker_ai.status_machine.state_machine_instance import StateMachineInstanceBuilder, ActionRegister
from thinker_ai.status_machine.state_machine_instance_repository import DefaultStateMachineContextRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import StartAction, MiddleAction


class TestSimpleStateMachine(unittest.TestCase):

    def setUp(self):
        ActionRegister.register_action(StartAction.get_full_class_name(), StartAction)
        ActionRegister.register_action(MiddleAction.get_full_class_name(), MiddleAction)
        self.base_dir = os.path.dirname(__file__)
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      'test_simple_state_machine_definitions.json')
        self.instance_repo = DefaultStateMachineContextRepository.from_file(self.base_dir,
                                                                            'test_simple_state_machine_instances.json',
                                                                            StateMachineInstanceBuilder(),
                                                                            self.definition_repo)
        # 读取状态机实例
        self.state_machine = self.instance_repo.get("group_instance","simple_instance")

    def tearDown(self):
        self.state_machine = None
        self.state_machine = None

    def test_simple_state_machine(self):
        # Transition to middle state
        commands = self.state_machine.get_state_machine_def().get_self_validate_commands_in_order()
        self.state_machine.handle(commands[0][0])
        self.assertEqual(self.state_machine.current_state_context_des.state_def.name, "middle")
        # Transition to end state
        result = self.state_machine.handle(commands[0][1])
        self.assertIsNotNone(result.event)
        self.assertEqual(result.event.name, "middle_command_handled")
        self.assertEqual(self.state_machine.current_state_context_des.state_def.name, "end")
        self.assertEqual(self.state_machine.last_state().state_def.name, "middle")


if __name__ == '__main__':
    unittest.main()
