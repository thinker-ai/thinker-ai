from thinker_ai.status_machine.state_machine import (Action, Command, Event, CompositeAction, StateContext,
                                                     CompositeStateContext, ActionResult)


class MockAction(Action):

    def __init__(self, on_command: str, event: str):
        super().__init__(on_command)
        self.result = ActionResult(success=True, event=Event(id=on_command, name=event))

    def handle(self, command: Command, owner_state_context: "StateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            return self.result
        return ActionResult(success=False)


class MockCompositeAction(MockAction):
    def __init__(self, on_command: str, event: str):
        super().__init__(on_command, event)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            # inner_commands:List=owner_state_context.get_state_machine().get_state_machine_def().get_state_execute_plan()
            # for inner_command in inner_commands:
            #     owner_state_context.handle_inner(inner_command)
            return self.result
        return ActionResult(success=False)


class StartAction(MockAction):
    def __init__(self, on_command: str):
        super().__init__(on_command, f"{on_command}_handled")


class MiddleAction(MockAction):

    def __init__(self, on_command: str):
        super().__init__(on_command, f"{on_command}_handled")


class InnerStartAction(MockAction):
    def __init__(self, on_command: str):
        super().__init__(on_command, f"{on_command}_handled")


class MiddleStartAction(CompositeAction):
    def __init__(self, on_command: str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_command = Command(name="inner_start_command", target="middle_start_instance")
            return owner_state_context.handle_inner(inner_command)
        return ActionResult(success=False)


class OuterStartAction(CompositeAction):
    def __init__(self, on_command: str):
        super().__init__(on_command)

    def handle(self, command: Command, owner_state_context: "CompositeStateContext", **kwargs) -> ActionResult:
        if command.name == self.on_command:
            inner_command = Command(name="middle_start_command", target="outer_start_instance")
            return owner_state_context.handle_inner(inner_command)
        return ActionResult(success=False)
