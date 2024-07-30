import os
import unittest
from thinker_ai.status_machine.state_machine import Command, ActionFactory
from thinker_ai.status_machine.state_machine_repository import FileBasedStateMachineContextRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import StartAction,MiddleAction


class TestSimpleStateMachine(unittest.TestCase):

    def setUp(self):
        ActionFactory.register_action(StartAction.get_full_class_name(), StartAction)
        ActionFactory.register_action(MiddleAction.get_full_class_name(), MiddleAction)
        self.base_dir = os.path.dirname(__file__)
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                         'test_simple_state_machine_definitions.json')
        self.instance_repo = FileBasedStateMachineContextRepository.from_file(self.base_dir,
                                                                    'test_simple_state_machine_instances.json',
                                                                    self.definition_repo)

        # 读取状态机实例
        self.state_machine = self.instance_repo.get("simple_instance")

    def tearDown(self):
        self.state_machine = None
        self.state_machine = None

    def test_simple_state_machine(self):
        # Transition to middle state
        self.state_machine.handle(Command(name="start_command", target=self.state_machine.id))
        self.assertEqual(self.state_machine.current_state_context.state_def.name, "middle")

        # Transition to end state
        command = Command(name="middle_command", target=self.state_machine.id)
        result = self.state_machine.handle(command)

        self.assertIsNotNone(result.event)
        self.assertEqual(result.event.name, "middle_command_handled")
        self.assertEqual(self.state_machine.current_state_context.state_def.name, "end")
        self.assertEqual(self.state_machine.last_state().state_def.name, "middle")


if __name__ == '__main__':
    unittest.main()
