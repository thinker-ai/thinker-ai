from typing import Optional

from thinker_ai.status_machine.state_machine import (Action, Command, Event, CompositeAction,StateContext,
                                                     CompositeStateContext)


class StartAction(Action):

    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class MiddleAction(Action):

    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class InnerStartAction(Action):

    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            return Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
        return None


class MiddleStartAction(CompositeAction):
    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            inner_command = Command(name="inner_start_command", target="middle_start_instance")
            return owner_state_context.handle_inner(inner_command)
        return None


class OuterStartAction(CompositeAction):
    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> Optional[Event]:
        if command.name == self.on_command:
            inner_command = Command(name="middle_start_command", target="outer_start_instance")
            event = owner_state_context.handle_inner(inner_command)
            if event:
                return event
        return None
