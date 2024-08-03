from typing import Type, Optional

from thinker_ai.agent.actions.di.task_tree import AskPlanResult
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.common.logs import logger
from thinker_ai.status_machine.state_machine_definition import StateDefinition, StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import CompositeStateScenario, StateScenarioBuilder, \
    StateMachineRepository


class StateMachineTaskScenario(CompositeStateScenario):

    def __init__(self, scenario_id: str,
                 scenario_group_id: str,
                 state_def: StateDefinition,
                 state_scenario_builder_class: Type[StateScenarioBuilder],
                 state_machine_scenario_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository,
                 plan_update_max_retry: int = 3,
                 ):
        super().__init__(scenario_id, scenario_group_id, state_def, state_scenario_builder_class,
                         state_machine_scenario_repository, state_machine_definition_repository)
        self.plan_update_max_retry = plan_update_max_retry
        self.exec_logger: Memory = Memory()

    async def plan_and_act(self, plan_update_max_retry: Optional[int] = None,
                           task_execute_max_retry: Optional[int] = None):
        if await self.try_plan(plan_update_max_retry):
            execute_plan_success = await self.try_act(task_execute_max_retry)
            if execute_plan_success:
                self.task_result = await AskPlanResult().run(self)
            else:
                logger.info("计划执行失败")

    async def try_plan(self, max_retry) -> bool:
        max_retry = max_retry if max_retry else self.plan_update_max_retry
        plan_update_count = 0
        while plan_update_count < max_retry:
            plan_update_count += 1
            if await self.write_plan():
                return True
        logger.info("更新计划次数超限，任务失败")
        return False

    async def try_act(self, max_retry: Optional[int] = None) -> bool:
        max_retry = max_retry if max_retry else self.task_execute_max_retry
        current_task_retry_counter = 0
        while self.current_task and current_task_retry_counter < max_retry:
            current_task_retry_counter += 1
            await self._check_data()
            await self.current_task.write_and_exec_code(first_trial=current_task_retry_counter == 1)
            current_result = self.current_task.task_result
            pass_review = False
            if current_result and current_result.is_success:
                pass_review = await self.review_current_result()
            if pass_review:
                self._update_current_task()
                current_task_retry_counter = 0

        if len(self.tasks) == len(self.get_finished_tasks()):
            return True
        else:
            return False
