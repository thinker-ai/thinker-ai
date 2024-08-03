from thinker_ai.status_machine.state_machine_scenario import MockExecutor, MockCompositeExecutor


class StartExecutor(MockExecutor):
    pass


class MiddleExecutor(MockExecutor):
    pass


class InnerStartExecutor(MockExecutor):
    pass


class MiddleStartAction(MockCompositeExecutor):
    pass


class OuterStartAction(MockCompositeExecutor):
    pass
