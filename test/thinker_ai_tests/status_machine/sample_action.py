from typing import Optional, Dict

from thinker_ai.status_machine.state_machine import Action, Command, Event, StateMachineContext


class SampleAction(Action):
    def __init__(self, name: str):
        super().__init__(name)

    def handle(self, command: Command, outer_state_machine: StateMachineContext, **kwargs) -> Optional[Event]:
        if command.name == self.command:
            return Event(name=f"{self.command}_handled", id=command.id, payload=command.payload)
        return None

    @classmethod
    def from_dict(cls, data: Dict) -> 'SampleAction':
        return cls(name=data['command'])



