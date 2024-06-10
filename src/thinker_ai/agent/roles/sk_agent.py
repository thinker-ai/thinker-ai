from typing import Any, Callable, Union

from pydantic import Field
from semantic_kernel import Kernel
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning.action_planner.action_planner import ActionPlanner
from semantic_kernel.planning.basic_planner import BasicPlanner, Plan

from thinker_ai.agent.actions import UserRequirement
from thinker_ai.agent.provider.schema import Message
from thinker_ai.agent.actions.execute_task import ExecuteTask
from thinker_ai.agent.roles import Role
from thinker_ai.common.logs import logger
from thinker_ai.utils.make_sk_kernel import make_sk_kernel


class SkAgent(Role):
    """
    Represents an SkAgent implemented using semantic kernel

    Attributes:
        name (str): Name of the SkAgent.
        profile (str): Role profile, default is 'sk_agent'.
        goal (str): Goal of the SkAgent.
        constraints (str): Constraints for the SkAgent.
    """

    name: str = "Sunshine"
    profile: str = "sk_agent"
    goal: str = "Execute task based on passed in task description"
    constraints: str = ""

    plan: Plan = Field(default=None, exclude=True)
    planner_cls: Any = None
    planner: Union[BasicPlanner, SequentialPlanner, ActionPlanner] = None
    kernel: Kernel = Field(default_factory=Kernel)
    import_semantic_skill_from_directory: Callable = Field(default=None, exclude=True)
    import_skill: Callable = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        """Initializes the Engineer role with given attributes."""
        super().__init__(**data)
        self.set_actions([ExecuteTask()])
        self._watch([UserRequirement])
        self.kernel = make_sk_kernel()

        # how funny the interface is inconsistent
        if self.planner_cls == BasicPlanner or self.planner_cls is None:
            self.planner = BasicPlanner()
        elif self.planner_cls in [SequentialPlanner, ActionPlanner]:
            self.planner = self.planner_cls(self.kernel)
        else:
            raise Exception(f"Unsupported planner of type {self.planner_cls}")

        self.import_semantic_skill_from_directory = self.kernel.import_semantic_skill_from_directory
        self.import_skill = self.kernel.import_skill

    async def _think(self) -> None:
        self._set_state(0)
        # how funny the interface is inconsistent
        if isinstance(self.planner, BasicPlanner):
            self.plan = await self.planner.create_plan_async(self.rc.important_memory[-1].content, self.kernel)
            logger.info(self.plan.generated_plan)
        elif any(isinstance(self.planner, cls) for cls in [SequentialPlanner, ActionPlanner]):
            self.plan = await self.planner.create_plan_async(self.rc.important_memory[-1].content)

    async def _act(self) -> Message:
        # how funny the interface is inconsistent
        result = None
        if isinstance(self.planner, BasicPlanner):
            result = await self.planner.execute_plan_async(self.plan, self.kernel)
        elif any(isinstance(self.planner, cls) for cls in [SequentialPlanner, ActionPlanner]):
            result = (await self.plan.invoke_async()).result
        logger.info(result)

        msg = Message(content=result, role=self.profile, cause_by=self.rc.todo)
        self.rc.memory.add(msg)
        return msg
