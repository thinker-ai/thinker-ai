import unittest
from typing import Dict

from thinker_ai.status_machine.state_machine import StateMachine, State, StateMachineDefinition, ActionFactory
from thinker_ai.status_machine.status_machine_repository import StateMachineDefinitionAbstractPersistence, \
    StateMachineAbstractPersistence, StateMachineDefinitionRepository, StateMachineRepository
from thinker_ai_tests.status_machine.test_status_machine import TestAction


# Assuming you have imported everything needed above


class MockStateMachineDefinitionPersistence(StateMachineDefinitionAbstractPersistence):
    def __init__(self):
        self.data_store: Dict[str, str] = {}

    def save(self, definition_id: str, data: str) -> None:
        self.data_store[definition_id] = data

    def load(self, definition_id: str) -> str:
        return self.data_store.get(definition_id, "")


class MockStateMachinePersistence(StateMachineAbstractPersistence):
    def __init__(self):
        self.data_store: Dict[str, Dict[str, str]] = {}

    def save(self, definition_id: str, instance_id: str, data: str) -> None:
        self.data_store.setdefault(definition_id, {})[instance_id] = data

    def load(self, definition_id: str, instance_id: str) -> str:
        return self.data_store.get(definition_id, {}).get(instance_id, "")


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.definition_persistence = MockStateMachineDefinitionPersistence()
        self.instance_persistence = MockStateMachinePersistence()

        self.definition_repo = StateMachineDefinitionRepository(self.definition_persistence)
        self.instance_repo = StateMachineRepository(self.instance_persistence, self.definition_repo)

        ActionFactory.register_action("test_action", TestAction)

    def test_save_and_load_definition(self):
        definition = StateMachineDefinition(business_id="test_business", states={}, transitions={})
        self.definition_repo.save(definition)
        loaded_definition = self.definition_repo.load("test_business")
        self.assertEqual(loaded_definition.business_id, "test_business")

    def test_save_and_load_instance(self):
        ActionFactory.register_action("test_command", TestAction)
        state = State(name="start", actions={"test_command": TestAction()})
        definition = StateMachineDefinition(business_id="test_business", states={"start": state}, transitions={})
        self.definition_repo.save(definition)

        sm = StateMachine(definition=definition, instance_id="1", current_state=state, history=[])
        self.instance_repo.save(sm)

        loaded_sm = self.instance_repo.load("test_business", "1")
        self.assertEqual(loaded_sm.current_state.name, "start")
        self.assertEqual(loaded_sm.instance_id, "1")

    def test_missing_definition(self):
        with self.assertRaises(ValueError):
            self.definition_repo.load("missing_business")

    def test_missing_instance(self):
        state = State(name="start", actions={"test_command": TestAction()})
        definition = StateMachineDefinition(business_id="test_business", states={"start": state}, transitions={})
        self.definition_repo.save(definition)

        with self.assertRaises(ValueError):
            self.instance_repo.load("test_business", "missing_instance")

    # You can add more tests as needed


if __name__ == "__main__":
    unittest.main()
