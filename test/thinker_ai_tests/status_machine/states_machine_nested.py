import unittest
import os

from thinker_ai.status_machine.state_machine import Command, ActionFactory
from thinker_ai.status_machine.state_machine_repository import FileBasedStateMachineContextRepository
from thinker_ai.status_machine.status_machine_definition_repository import FileBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import InnerStartAction,MiddleStartAction,OuterStartAction


class TestNestedStateMachine(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'test_nested_state_machine_definitions.json'
        self.instances_file_name = 'test_nested_state_machine_instances.json'
        self.definition_repo = FileBasedStateMachineDefinitionRepository(self.base_dir, self.definitions_file_name)
        self.instance_repo = FileBasedStateMachineContextRepository(self.base_dir, self.instances_file_name,
                                                                    self.definition_repo)
        ActionFactory.register_action(InnerStartAction.get_full_class_name(), InnerStartAction)
        ActionFactory.register_action(MiddleStartAction.get_full_class_name(), MiddleStartAction)
        ActionFactory.register_action(OuterStartAction.get_full_class_name(), OuterStartAction)
        # 读取状态机实例
        self.state_machine = self.instance_repo.load("outer_instance")

    def tearDown(self):
        self.state_machine = None

    def test_combined_state_machine(self):
        command = Command(name="outer_start_command", target=self.state_machine.id)
        self.state_machine.handle(command)
        # Verify middle state machine transitions
        middle_state_machine = self.instance_repo.load("outer_start_instance")
        self.assertEqual("middle_end",middle_state_machine.current_state_context.state_def.name)
        self.assertEqual("middle_start",middle_state_machine.last_state().state_def.name)
        # Check if the outer state machine has also transitioned
        self.assertEqual("outer_end",self.state_machine.current_state_context.state_def.name)
        self.assertEqual("outer_start",self.state_machine.last_state().state_def.name)


if __name__ == '__main__':
    unittest.main()