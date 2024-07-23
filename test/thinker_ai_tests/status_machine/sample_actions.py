from typing import Optional

from thinker_ai.status_machine.state_machine import (Action, Command, Event, CompositeAction,StateContext,
                                                     CompositeStateContext)


class StartAction(Action):

    def __init__(self):
        super().__init__("start_command")

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class MiddleAction(Action):

    def __init__(self):
        super().__init__("middle_command")

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class InnerStartAction(Action):

    def __init__(self):
        super().__init__("inner_start_command")

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class MiddleStartAction(CompositeAction):
    def __init__(self):
        super().__init__("middle_start_command")

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            inner_command = Command(name="inner_start_command", target="inner_instance")
            inner_event = self._handle_single_inner_command(inner_command,owner_state_context)
            if inner_event is not None:
                return self.to_outer_event(inner_event, owner_state_context)
        return None


class OuterStartAction(CompositeAction):
    def __init__(self):
        super().__init__("outer_start_command")

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            middle_command = Command(name="middle_start_command", target="middle_instance")
            middle_event = self._handle_single_inner_command(middle_command,owner_state_context)
            if middle_event is not None:
                return self.to_outer_event(middle_event, owner_state_context)
        return None
