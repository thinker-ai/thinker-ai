import json
import os
from typing import Optional, Dict

from thinker_ai.status_machine.state_machine import Action, Command, Event, ActionFactory
from thinker_ai.status_machine.state_machine_builder import create_state_machine


def read_json_file(file_path):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, file_path)
    with open(full_path, 'r') as file:
        return json.load(file)


def construct_state_machine(file_path):
    json_data = read_json_file(file_path)
    return create_state_machine(json_data)


class SampleAction(Action):
    def __init__(self, name: str):
        super().__init__(name)

    def handle(self, command: Command) -> Optional[Event]:
        if command.name == self.name:
            return Event(name=f"{self.name}_handled", id=command.id, payload=command.payload)
        return None

    @classmethod
    def from_dict(cls, data: Dict) -> 'SampleAction':
        return cls(name=data['name'])


ActionFactory.register_action('SampleAction', SampleAction)
