from thinker_ai.status_machine.state_machine_context import MockAction, MockCompositeAction


class StartAction(MockAction):
    pass


class MiddleAction(MockAction):
    pass


class InnerStartAction(MockAction):
    pass


class MiddleStartAction(MockCompositeAction):
    pass


class OuterStartAction(MockCompositeAction):
    pass
