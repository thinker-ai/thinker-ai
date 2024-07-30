from thinker_ai.status_machine.state_machine import (Action, Command, Event, CompositeAction, StateContext,
                                                     CompositeStateContext, ActionResult)


class StartAction(Action):

    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            event = Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
            return ActionResult(success=True,event=event)
        return ActionResult(success=False)


class MiddleAction(Action):

    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            event = Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
            return ActionResult(success=True,event=event)
        return ActionResult(success=False)


class InnerStartAction(Action):
    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            event = Event(name=f"{self.on_command}_handled", id=command.id, payload=command.payload)
            return ActionResult(success=True,event=event)
        return ActionResult(success=False)


class MiddleStartAction(CompositeAction):
    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_command = Command(name="inner_start_command", target="middle_start_instance")
            return owner_state_context.handle_inner(inner_command)
        return ActionResult(success=False)


class OuterStartAction(CompositeAction):
    def __init__(self,on_command:str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_command = Command(name="middle_start_command", target="outer_start_instance")
            return owner_state_context.handle_inner(inner_command)
        return ActionResult(success=False)
