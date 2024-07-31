from thinker_ai.status_machine.state_machine import (Action, Command, Event, CompositeAction, StateContext,
                                                     CompositeStateContext, ActionResult)


class MockAction(Action):

    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            result = ActionResult(success=True, event=Event(id=self.on_command, name=command.payload.get("event")))
            return result
        return ActionResult(success=False)


class MockCompositeAction(CompositeAction):
    def __init__(self, on_command):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_commands = (owner_state_context.get_state_machine()
                              .get_state_machine_def().get_validate_command_in_order())
            for inner_command in inner_commands:
                owner_state_context.handle_inner(inner_command)
            result = ActionResult(success=True, event=Event(id=self.on_command, name=command.payload.get("event")))
            return result
        return ActionResult(success=False)


class StartAction(MockAction):
    def __init__(self, on_command):
        super().__init__(on_command)


class MiddleAction(MockAction):
    def __init__(self, on_command):
        super().__init__(on_command)


class InnerStartAction(MockAction):
    def __init__(self, on_command):
        super().__init__(on_command)


class MiddleStartAction(MockCompositeAction):
    def __init__(self, on_command):
        super().__init__(on_command)


class OuterStartAction(MockCompositeAction):
    def __init__(self, on_command):
        super().__init__(on_command)
