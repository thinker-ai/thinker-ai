import unittest
import os

from thinker_ai.status_machine.state_machine_context import StateMachineContextBuilder, ActionRegister
from thinker_ai.status_machine.state_machine_context_repository import DefaultStateMachineContextRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai_tests.status_machine.sample_actions import InnerStartAction, MiddleStartAction, OuterStartAction


class TestNestedStateMachine(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.definitions_file_name = 'test_nested_state_machine_definitions.json'
        self.contexts_file_name = 'test_nested_state_machine_contexts.json'
        self.definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(self.base_dir,
                                                                                      self.definitions_file_name)
        self.context_repo = DefaultStateMachineContextRepository.from_file(self.base_dir, self.contexts_file_name,
                                                                            StateMachineContextBuilder(),
                                                                            self.definition_repo)
        ActionRegister.register_action(InnerStartAction.get_full_class_name(), InnerStartAction)
        ActionRegister.register_action(MiddleStartAction.get_full_class_name(), MiddleStartAction)
        ActionRegister.register_action(OuterStartAction.get_full_class_name(), OuterStartAction)
        # 读取状态机实例
        self.state_machine = self.context_repo.get("group_context","outer_context")

    def tearDown(self):
        self.state_machine = None

    def test_combined_state_machine(self):
        commands = self.state_machine.get_state_machine_def().get_self_validate_commands_in_order()
        self.state_machine.handle(commands[0][0])
        # Verify middle state machine transitions
        middle_state_machine = self.context_repo.get("group_context","outer_start_context")
        self.assertEqual("middle_end", middle_state_machine.current_state_context_des.state_def.name)
        self.assertEqual("middle_start", middle_state_machine.last_state().state_def.name)
        # Check if the outer state machine has also transitioned
        self.assertEqual("outer_end", self.state_machine.current_state_context_des.state_def.name)
        self.assertEqual("outer_start", self.state_machine.last_state().state_def.name)


if __name__ == '__main__':
    unittest.main()
